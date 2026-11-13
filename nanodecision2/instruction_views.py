"""Build explicitly reviewed instruction pairs for the existing JS training path.

Only training rows are augmented. Review metadata is an assertion by the data
owner, not an automatic proof of semantic equivalence. No teacher API is used.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from .schema import normalize_question, gold_target
from .runtime import sha256

_ALLOWED = {'view_id', 'reviewed', 'question', 'instructions', 'criteria', 'choices',
            'language', 'state_language', 'question_language', 'candidate_languages', 'code_switch'}


def reviewed_pair(row, seed=42):
    """Select one audited question/candidate rewrite, preserving source and labels."""
    if row.get('partition') != 'train':
        raise ValueError('Instruction views require an explicit train partition')
    if not row.get('id') or not row.get('group_id'):
        raise ValueError('id and original group_id are required')
    if row.get('equivalence_verified') or row.get('equivalence_id'):
        raise ValueError('Do not re-augment an existing equivalence pair')
    views = row.get('instruction_views')
    if not isinstance(views, list) or not views:
        raise ValueError('No reviewed instruction_views supplied')
    if any(not isinstance(v, dict) or v.get('reviewed') is not True or
           not isinstance(v.get('view_id'), str) or not v['view_id'] or set(v)-_ALLOWED for v in views):
        raise ValueError('Each view needs a unique view_id, reviewed=true, and allowed input fields')
    ids = [v['view_id'] for v in views]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate view IDs')
    views = sorted(views, key=lambda v: v['view_id'])
    pick = int(hashlib.sha256(f'{seed}:{row["id"]}'.encode()).hexdigest(), 16) % len(views)
    view = views[pick]
    original = copy.deepcopy(row)
    original.pop('instruction_views')
    changed = copy.deepcopy(original)
    if 'question' in view or 'instructions' in view:
        changed.pop('question', None); changed.pop('instructions', None)
    if 'criteria' in view or 'choices' in view:
        changed.pop('criteria', None); changed.pop('choices', None)
    changed.update({k: copy.deepcopy(v) for k, v in view.items() if k not in {'view_id','reviewed'}})
    a, b = normalize_question(original), normalize_question(changed)
    if a == b:
        raise ValueError('A repeated prompt is not a distinct instruction view')
    ia, ib = [o['id'] for o in a['options']], [o['id'] for o in b['options']]
    if a['type'] != b['type'] or set(ia) != set(ib) or a['values'] != b['values']:
        raise ValueError('Equivalent views must preserve candidate IDs/type/score levels')
    target = dict(zip(ia, gold_target(original, a)))
    changed['target_probs'] = [target[i] for i in ib]
    gold_target(changed, b)
    key = hashlib.sha256(f'{seed}:{row["group_id"]}:{row["id"]}:{view["view_id"]}'.encode()).hexdigest()
    for item in (original, changed):
        item.update(equivalence_id='instruction:'+key, equivalence_verified=True,
                    equivalence_review='dataset_author_asserted', original_example_id=row['id'])
    changed['id'] = row['id'] + ':view:' + view['view_id']
    return original, changed


def build_views(source, output, fraction=.25, seed=42):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be between zero and one')
    source, output = Path(source), Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.partial')
    counts = {'source_rows':0,'output_rows':0,'reviewed_pairs':0}
    try:
        with source.open(encoding='utf-8') as src, temporary.open('x',encoding='utf-8') as dst:
            for line in src:
                if not line.strip():continue
                row=json.loads(line);counts['source_rows']+=1
                h=int(hashlib.sha256(f'{seed}:{row["id"]}:include'.encode()).hexdigest()[:16],16)/2**64
                if row.get('partition')=='train' and row.get('instruction_views') and h < fraction:
                    batch=reviewed_pair(row,seed);counts['reviewed_pairs']+=1
                else:
                    row.pop('instruction_views',None);batch=[row]
                for item in batch:
                    dst.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');counts['output_rows']+=1
        temporary.replace(output)
    except BaseException:
        # Keep an existing/failed partial distinguishable; never publish it as complete.
        raise
    report={**counts,'source_sha256':sha256(source),'output_sha256':sha256(output),
        'seed':seed,'fraction':fraction,'note':'Views are not independent source examples. Semantics are asserted by the data reviewer.'}
    output.with_suffix(output.suffix+'.manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fraction',type=float,default=.25);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(argv);print(json.dumps(build_views(a.input,a.out,a.fraction,a.seed),indent=2))

if __name__=='__main__':main()
