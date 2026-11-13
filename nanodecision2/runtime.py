"""Rust tokenizer + parallel C++ network; no torch/numpy imports at runtime."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from .schema import TextCodec, normalize_question
from .probability import softmax, prediction_set, bayes_action
from .multilingual import served_temperature
from .decision_contract import API_CONTRACT_VERSION, calibration_language, decision_result


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class DecisionEngine:
    def __init__(self, bundle, require_calibration=None, *, legacy_policy=False,
                 allow_smoke=False):
        from . import _native
        self.native = _native
        self.root = Path(bundle)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.cfg = self.manifest['config']
        self.legacy_policy = bool(legacy_policy)
        self.require_calibration = self.legacy_policy if require_calibration is None else bool(require_calibration)
        self.smoke_only = bool(self.manifest.get('smoke_only', False))
        if self.smoke_only and not allow_smoke:
            raise ValueError('Test-only model: pass allow_smoke=True explicitly for fixture execution')
        actual_hash = sha256(self.root / 'model.nd2')
        if self.manifest['model_sha256'] != actual_hash:
            raise ValueError('Model file hash does not match manifest')
        expected_tokenizer = self.manifest.get('provenance', {}).get('tokenizer_sha256')
        if not self.smoke_only and expected_tokenizer:
            if sha256(self.root / 'tokenizer.json') != expected_tokenizer:
                raise ValueError('Tokenizer file hash does not match model provenance')
        self.calibration = None
        path = self.root / 'calibration.json'
        if path.exists():
            self.calibration = json.loads(path.read_text(encoding='utf-8'))
            if self.calibration['model_sha256'] != actual_hash:
                raise ValueError('Calibration belongs to another model; recalibrate after quantization')
            # Validate the global temperature even when a request uses a bucket.
            softmax([0.0, 0.0], self.calibration['temperature'])
            if 'qhat' in self.calibration:
                prediction_set([0.5, 0.5], self.calibration['qhat'])
        self.handle = _native.load(str(self.root / 'model.nd2'))
        self.codec = TextCodec(self.root / 'tokenizer.json', smoke=self.smoke_only)
        # Per-engine bounded token cache. Model embeddings are cached in C++.
        self._pair_cache = lru_cache(maxsize=256)(self._encode_pairs)

    def _encode_pairs(self, serialized_spec):
        spec = json.loads(serialized_spec)
        return tuple(tuple(ids) for ids in self.codec.pairs(spec, self.manifest['max_pair_tokens']))

    def prepare(self, state):
        ids = self.codec.state(state, self.manifest['max_state_tokens'])
        return self.native.prepare(self.handle, ids)

    def score_prepared(self, prepared, question, min_confidence=0.0, language=None,
                       language_metadata=None):
        if not 0 <= min_confidence <= 1:
            raise ValueError('min_confidence must lie in [0,1]')
        spec = normalize_question(question)
        key = json.dumps(spec, ensure_ascii=False, sort_keys=True, allow_nan=False)
        pairs = self._pair_cache(key)
        logits = self.native.score(self.handle, prepared, pairs)
        lang = calibration_language(question, language, language_metadata)
        temperature = served_temperature(self.calibration, lang, spec['type'], len(spec['options']))
        p = softmax(logits, temperature)
        result = decision_result(spec, p)
        indices = None
        if self.calibration and 'qhat' in self.calibration:
            indices = prediction_set(p, self.calibration['qhat'])
        names = [o['id'] for o in spec['options']]
        reasons = []
        if self.require_calibration and self.calibration is None:
            reasons.append('uncalibrated')
        if result['top1_probability'] < min_confidence:
            reasons.append('low_confidence')
        # Reading an old qhat MUST NOT silently enable the old policy.
        if self.legacy_policy and indices is not None and len(indices) != 1:
            reasons.append('non_singleton_prediction_set')
        if self.legacy_policy and self.smoke_only:
            reasons.append('smoke_model_not_for_use')
        result.update(
            logits=logits, raw_probabilities=dict(zip(names, softmax(logits))),
            temperature_used=temperature, calibration_language=lang,
            calibrated=self.calibration is not None,
            calibration_status='calibrated' if self.calibration else 'raw',
            prediction_set=None if indices is None else [names[i] for i in indices],
            smoke_only=self.smoke_only,
            model_sha256=self.manifest['model_sha256'],
            matrix_storage=self.manifest.get('matrix_storage'),
            policy_result={'mode': 'legacy' if self.legacy_policy else 'advisory',
                           'review_recommended': bool(reasons), 'reasons': reasons},
        )
        if self.smoke_only:
            result['status'] = 'test_only'
        if self.legacy_policy:
            result['decision_mode'] = 'legacy_policy'
            result['value'] = None if reasons else result['value']
            result['abstained'] = bool(reasons)
            result['abstain_reasons'] = reasons
        return result

    def decide(self, state, questions, min_confidence=0.0, language=None,
               language_metadata=None):
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
            raise ValueError('questions must map 1..64 names to question specifications')
        ctx = self.prepare(state)
        return {
            'api_contract_version': API_CONTRACT_VERSION,
            'answers': {name: self.score_prepared(ctx, q, min_confidence, language,
                                                language_metadata)
                        for name, q in questions.items()},
            'model_sha256': self.manifest['model_sha256'],
        }

    def logits_row(self, row):
        return self.score_prepared(self.prepare(row['state']), row,
                                   language_metadata=row.get('language_metadata'))

    def clear_cache(self):
        self._pair_cache.cache_clear()
        self.native.clear_cache(self.handle)

    @staticmethod
    def select_action(probabilities, cost_matrix, abstain_cost=None):
        return bayes_action(probabilities, cost_matrix, abstain_cost)
