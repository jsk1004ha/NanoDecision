"""Version 2 decision contract. A model decision is not an execution policy.

Standard-library only: importing the runtime must not load training libraries.
"""
import math
from .probability import check_distribution
from .multilingual import language_key

API_CONTRACT_VERSION = 2


def calibration_language(question, default=None, metadata=None):
    """Use declared component metadata, never guess language from confidence.

    Per-question metadata wins over request defaults. Mixed component languages
    select the mixed bucket rather than averaging English/Korean temperatures.
    """
    declared = dict(metadata or {})
    declared.update({key: question[key] for key in (
        'state_language', 'question_language', 'candidate_languages', 'code_switch'
    ) if key in question})
    if declared.get('code_switch') is True:
        return 'mixed'
    values = [declared.get('state_language'), declared.get('question_language')]
    candidates = declared.get('candidate_languages', [])
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        raise ValueError('candidate_languages must be a list or ID-language mapping')
    languages = {language_key(x) for x in values + list(candidates)} - {'unknown'}
    if len(languages) > 1 or 'mixed' in languages:
        return 'mixed'
    if languages:
        return next(iter(languages))
    return language_key(question.get('language') or default)


def decision_result(spec, probabilities):
    """Return a non-null result for every valid distribution, including ties.

    Exact ties use lexicographic candidate IDs; IDs are not neural inputs.
    Score.value remains the expectation for compatibility, not the modal level.
    """
    check_distribution(probabilities)
    names = [o['id'] for o in spec['options']]
    if len(names) != len(probabilities) or len(set(names)) != len(names):
        raise ValueError('Probability/candidate mismatch')
    top = max(probabilities)
    ties = sorted(name for name, p in zip(names, probabilities) if p == top)
    chosen = ties[0]
    index = names.index(chosen)
    value = chosen
    result = {
        'api_contract_version': API_CONTRACT_VERSION,
        'decision_mode': 'best_candidate', 'status': 'ok',
        'type': spec['type'], 'selected_option_id': chosen,
        'probabilities': dict(zip(names, probabilities)),
        'top1_probability': top, 'confidence': top,
        'tied_candidate_ids': ties if len(ties) > 1 else [],
    }
    if spec['type'] == 'noul':
        if set(names) != {'false', 'true'}:
            raise ValueError('Noul requires false/true candidates')
        value = chosen == 'true'
        result['noul'] = probabilities[names.index('true')]
    elif spec['type'] == 'score':
        values = spec['values']
        value = math.fsum(p * v for p, v in zip(probabilities, values))
        if not math.isfinite(value):
            raise ValueError('Non-finite score expectation')
        result.update(mode_value=values[index], expected_value=value)
    result.update(value=value, suggested_value=value, abstained=False, abstain_reasons=[])
    return result
