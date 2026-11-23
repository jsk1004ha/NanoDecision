"""Retrieve from a large choice catalog, then rerank a bounded shortlist.

Request fields are state, utterance, question, criteria, and optional language.
Retrieval sees only visible state, utterance, and candidate descriptions. The
model state is always {"context": state, "utterance": utterance}; training for
this path must use the same representation. When retrieval truncates the
catalog, returned probabilities are conditional on that shortlist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata

from .runtime import DecisionEngine
from .schema import state_text


METHOD = "unicode_char_3_5_plus_word_tfidf_cosine_v1"
ALLOWED_REQUEST_KEYS = {"state", "utterance", "question", "criteria", "language"}


def _text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _ngrams(value):
    value = _text(value)
    terms = [value[start:start + size]
             for size in range(3, 6)
             for start in range(max(0, len(value) - size + 1))]
    # Exact word features preserve short Korean and identifier-like terms whose
    # complete value is shorter than the character n-gram range.
    terms.extend("word:" + word for word in re.findall(r"[^\W_]+", value))
    return Counter(terms)


def _catalog(request):
    if not isinstance(request, dict):
        raise ValueError("Catalog request must be an object")
    unexpected = set(request) - ALLOWED_REQUEST_KEYS
    if unexpected:
        raise ValueError("Unexpected catalog request fields: " + ", ".join(sorted(unexpected)))
    if "state" not in request:
        raise ValueError("Catalog request requires visible state")
    utterance = request.get("utterance")
    question = request.get("question")
    criteria = request.get("criteria")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("Catalog request requires a nonempty utterance")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Catalog request requires a nonempty question")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError("Catalog criteria must map at least two IDs to descriptions")
    if not all(isinstance(key, str) and key.strip() and isinstance(description, str)
               and description.strip() for key, description in criteria.items()):
        raise ValueError("Catalog IDs and descriptions must be nonempty strings")
    descriptions = [_text(description) for description in criteria.values()]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("Duplicate catalog descriptions")
    language = request.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("language must be a nonempty string when supplied")
    try:
        visible_context = state_text(request["state"])
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON or text") from error
    return visible_context, utterance.strip(), question.strip(), criteria, language


def rank_catalog(state, utterance, criteria):
    """Return every catalog ID in deterministic relevance order."""
    documents = {key: _ngrams(description) for key, description in criteria.items()}
    document_frequency = Counter()
    for terms in documents.values():
        document_frequency.update(terms.keys())
    count = len(documents)
    inverse_frequency = {
        term: math.log((1 + count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    def vector(terms):
        weighted = {term: (1 + math.log(frequency)) * inverse_frequency[term]
                    for term, frequency in terms.items() if term in inverse_frequency}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return ({term: value / norm for term, value in weighted.items()} if norm else {})

    query = vector(_ngrams(state + "\n" + utterance))
    ranked = []
    for key, terms in documents.items():
        candidate = vector(terms)
        score = sum(value * query.get(term, 0.0) for term, value in candidate.items())
        ranked.append((key, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def decide_catalog(engine, request, top_k=32, min_confidence=0.0):
    """Retrieve from a validated full catalog and rerank it with DecisionEngine."""
    if type(top_k) is not int or not 2 <= top_k <= 256:
        raise ValueError("top_k must be an integer in [2,256]")
    state, utterance, question, criteria, language = _catalog(request)
    ranked = rank_catalog(state, utterance, criteria)
    shortlist_size = min(top_k, len(ranked))
    shortlist = ranked[:shortlist_size]
    boundary_score = shortlist[-1][1]
    boundary_ties = sum(score == boundary_score for _, score in ranked)
    short_criteria = {key: criteria[key] for key, _ in shortlist}
    model_state = {"context": request["state"], "utterance": utterance}
    result = engine.decide(
        model_state,
        {"catalog": {"type": "choice", "question": question, "criteria": short_criteria}},
        min_confidence=min_confidence,
        language=language,
    )
    answer = result["answers"]["catalog"]
    full_catalog = shortlist_size == len(ranked)
    temperature_calibrated = bool(answer.get("calibrated", False))
    temperature_calibration_status = answer.get("calibration_status", "unknown")
    answer.update(
        probability_scope=("full_catalog" if full_catalog
                           else "conditional_on_retrieved_shortlist"),
        temperature_scope=("full_catalog_candidate_count" if full_catalog
                           else "retrieved_shortlist_candidate_count"),
        calibration_scope=("full_catalog_candidate_count" if full_catalog
                           else "retrieved_shortlist_candidate_count"),
        full_catalog_probability_available=full_catalog,
        retrieval_coverage_calibrated=False,
    )
    if not full_catalog:
        answer.update(
            temperature_calibrated=temperature_calibrated,
            temperature_calibration_status=temperature_calibration_status,
            calibrated=False,
            calibration_status="uncalibrated_retrieval_coverage",
        )
        policy_requested = min_confidence > 0 or bool(getattr(engine, "legacy_policy", False))
        if policy_requested:
            reason = "retrieval_coverage_uncalibrated"
            policy = answer.setdefault(
                "policy_result",
                {"mode": "legacy" if getattr(engine, "legacy_policy", False) else "advisory",
                 "review_recommended": False, "reasons": []},
            )
            if reason not in policy["reasons"]:
                policy["reasons"].append(reason)
            policy["review_recommended"] = True
            if getattr(engine, "legacy_policy", False):
                answer["value"] = None
                answer["abstained"] = True
                reasons = answer.setdefault("abstain_reasons", [])
                if reason not in reasons:
                    reasons.append(reason)
    result["retrieval"] = {
        "method": METHOD,
        "catalog_size": len(ranked),
        "requested_top_k": top_k,
        "shortlist_size": shortlist_size,
        "boundary_score": boundary_score,
        "boundary_tie_count": boundary_ties,
        "tie_break": "score_descending_then_candidate_id_ascending",
        "query_fields": ["state", "utterance"],
        "model_state_contract": {"context": "state", "utterance": "utterance"},
        "shortlist": [
            {"id": key, "rank": rank, "score": score}
            for rank, (key, score) in enumerate(shortlist, start=1)
        ],
    }
    return result


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nd2 predict-catalog")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=0)
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Legacy compatibility; raw probabilities are allowed by default")
    parser.add_argument("--legacy-policy", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"),
                             object_pairs_hook=_unique_object)
        engine = DecisionEngine(
            args.bundle,
            require_calibration=args.legacy_policy and not args.allow_uncalibrated,
            legacy_policy=args.legacy_policy,
            allow_smoke=args.allow_smoke,
        )
        result = decide_catalog(engine, payload, args.top_k, args.min_confidence)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
