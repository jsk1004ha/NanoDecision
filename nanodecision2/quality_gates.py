"""Whole-workload metrics and explicit per-language/per-task regression gates."""
import math
from collections import defaultdict
from .decision_contract import decision_result
from .probability import softmax, metrics
from .multilingual import language_key


class QualityReport:
    def __init__(self):
        self.cells = defaultdict(lambda: {'count': 0, 'hard_count': 0, 'correct': 0, 'nll': 0., 'brier': 0.})

    def add(self, row, logits):
        p = softmax(logits[:len(row['target_probs'])])
        m = metrics(p, row['target_probs'], row.get('values'))
        names = row['option_ids']
        predicted = decision_result({'type': row['type'], 'options': [{'id': k} for k in names],
                                     'values': row.get('values')}, p)['selected_option_id']
        hard = max(row['target_probs']) > 1 - 1e-7
        correct = hard and predicted == names[row['target_probs'].index(max(row['target_probs']))]
        lang, task = language_key(row.get('language')), str(row.get('task', row['type']))
        for key in ('all', 'language:' + lang, 'task:' + task, 'slice:' + lang + ':' + task):
            c = self.cells[key]; c['count'] += 1; c['hard_count'] += hard; c['correct'] += correct
            c['nll'] += m['nll']; c['brier'] += m['brier_to_target']

    def summary(self):
        if not self.cells:
            raise ValueError('Empty evaluation')
        cells = {k: {**c, 'accuracy': c['correct']/c['hard_count'] if c['hard_count'] else None,
                     'nll': c['nll']/c['count'], 'brier': c['brier']/c['count']} for k, c in self.cells.items()}
        tasks = [c['accuracy'] for k, c in cells.items() if k.startswith('task:') and c['accuracy'] is not None]
        return {'overall': cells['all'], 'cells': cells,
                'macro_task_accuracy': sum(tasks)/len(tasks) if tasks else None,
                'denominator': 'all supplied valid rows; no confidence filtering'}


def protected_slices(candidate, reference, languages=('en',), tolerance=1e-10):
    """No averaging away a regression in another task; tolerances are numeric only."""
    failures = []
    for language in languages:
        keys = [k for k in reference['cells'] if k.startswith('slice:' + language + ':')]
        if not keys:
            raise ValueError('Protected language has no validation tasks: ' + language)
        for key in keys:
            r, c = reference['cells'][key], candidate['cells'].get(key)
            if c is None or c['count'] != r['count'] or c['hard_count'] != r['hard_count']:
                failures.append({'slice': key, 'reason': 'missing_or_changed_population'}); continue
            if any(not math.isfinite(c[m]) or c[m] > r[m] + tolerance for m in ('nll', 'brier')):
                failures.append({'slice': key, 'reason': 'probability_regression'})
            if r['accuracy'] is not None and (c['accuracy'] is None or c['accuracy'] < r['accuracy'] - tolerance):
                failures.append({'slice': key, 'reason': 'accuracy_regression'})
    return {'passed': not failures, 'failures': failures, 'languages': list(languages)}


def selection_key(report):
    acc = report['macro_task_accuracy']
    if acc is None:
        raise ValueError('Accuracy-first selection needs hard-label validation tasks')
    return (-acc, report['overall']['nll'], report['overall']['brier'])
