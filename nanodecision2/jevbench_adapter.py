"""Local NanoDecision adapter for the public Benchmark Heaven JevBench harness.

Install a separately audited JevBench checkout. No remote model API is called;
its original Runner, scoring and summarize modules are used, not reimplemented.
"""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import time
from .decision_contract import decision_result
from .probability import check_distribution
from .runtime import DecisionEngine, sha256
from .auto_sources import atomic_json


def map_question(question, labels):
    q = dict(question)
    if len(labels) != len(set(labels)) or not all(isinstance(x,str) for x in labels):
        raise ValueError('Invalid canonical labels')
    kind = q.get('type')
    if kind == 'noul':
        if set(labels) != {'no','yes'}:
            raise ValueError('Canonical Noul labels must be no/yes')
        criteria = q.get('criteria')
        if criteria is not None:
            if not isinstance(criteria,dict) or set(criteria) not in ({'no','yes'},{'false','true'}):
                raise ValueError('Unsupported Noul criteria; refusing to rewrite its meaning')
            q['criteria'] = {'false':criteria['no'],'true':criteria['yes']} if set(criteria)=={'no','yes'} else dict(criteria)
        else:q.pop('criteria',None)
        return q, {'no':'false','yes':'true'}
    if kind == 'choice':
        if not isinstance(q.get('criteria'),dict) or set(q['criteria']) != set(labels):
            raise ValueError('Choice criteria must match every canonical label')
    elif kind == 'score':
        if not isinstance(q.get('criteria'),list) or labels != [str(i) for i in range(len(q['criteria']))]:
            raise ValueError('Score must preserve the complete ordered level inventory')
    else:raise ValueError('Unsupported canonical question type')
    return q, {label:label for label in labels}


class NanoDecisionAdapter:
    name = 'nanodecision-local'
    price_input_per_m = None
    price_output_per_m = None
    cost_basis = 'local_compute_unmetered'

    def __init__(self, bundle=None, *, engine=None, allow_smoke=False):
        self.engine = engine or DecisionEngine(bundle, allow_smoke=allow_smoke)
        self.model = 'nanodecision:' + self.engine.manifest['model_sha256']

    def reserve_estimate(self, task):
        return None

    def run(self, task):
        from jevbench.adapters.base import DecisionResult
        started = time.perf_counter()
        try:
            # The adapter never reads expected/provenance/group, nor retrieves
            # examples or changes candidate order based on benchmark labels.
            q, mapping = map_question(task.question,task.labels)
            answer = self.engine.decide(task.state,{'decision':q})['answers']['decision']
            probs = {name:answer['probabilities'][mapping[name]] for name in task.labels}
            check_distribution(list(probs.values()))
            return DecisionResult(self.name,True,probs=probs,probs_source='native',model=self.model,
                latency_s=time.perf_counter()-started,usage={},
                raw={'logits':answer.get('logits'),'runtime':{'backend':'native_cpu','model_sha256':self.engine.manifest['model_sha256'],
                     'temperature_used':answer.get('temperature_used'),'smoke_only':answer.get('smoke_only',False)}},
                request_body={'state':task.state,'questions':{'decision':q}})
        except (ValueError,KeyError,TypeError) as exc:
            # A capacity/schema failure remains an attempted wrong/invalid row.
            # Do not substitute uniform probabilities or remove the question.
            return DecisionResult(self.name,False,model=self.model,status=422,error=type(exc).__name__,
                                  latency_s=time.perf_counter()-started)


def run_public(bundle,tasks_path,out,upstream_commit,allow_smoke=False):
    import jevbench
    from jevbench.tasks import load_jsonl,dataset_hash
    from jevbench.runner import Runner
    from jevbench.budget import Ledger
    from jevbench.summarize import summarize
    import jevbench.scoring as scoring
    root = Path(jevbench.__file__).resolve().parents[1]
    actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual != upstream_commit:
        raise ValueError('JevBench checkout does not match --upstream-commit')
    # A modified scorer must never be described as the pinned upstream scorer.
    changed = subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True)
    if changed.strip():raise ValueError('JevBench checkout has modified tracked files')
    tasks = load_jsonl(str(tasks_path))
    if not tasks or any(t.split != 'public' for t in tasks):
        raise ValueError('This command accepts a nonempty public-only cohort')
    if len({t.id for t in tasks}) != len(tasks):raise ValueError('Duplicate benchmark IDs')
    out=Path(out).resolve()
    if out.is_relative_to(root):raise ValueError('Write raw benchmark evidence outside the upstream repository')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh evaluation output')
    out.mkdir(parents=True,exist_ok=True)
    adapter=NanoDecisionAdapter(bundle,allow_smoke=allow_smoke)
    manifest={'suite':'fstandhartinger/jevbench','cohort':'public-only','upstream_commit':actual,
        'task_file_sha256':sha256(tasks_path),'canonical_task_hash':dataset_hash(tasks),
        'scorer_sha256':sha256(scoring.__file__),'runner_sha256':sha256(inspect.getfile(Runner)),
        'summarizer_sha256':sha256(inspect.getfile(summarize)),
        'model_sha256':adapter.engine.manifest['model_sha256'],'planned':len(tasks),
        'official_full_score':None,'sealed_composite':None,'local_cost_usd':None,
        'scope':'Original serial harness, public-only, no remote API. No official ranking or sealed result.',
        'smoke_only':adapter.engine.manifest.get('smoke_only',False),
        'python':__import__('sys').version,'platform':__import__('platform').platform(),
        'matrix_storage':adapter.engine.manifest.get('matrix_storage'),
        'arithmetic':adapter.engine.manifest.get('arithmetic'),
        'tokenizer_sha256':adapter.engine.manifest.get('provenance',{}).get('tokenizer_sha256'),
        'thread_environment':{key:__import__('os').environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
    atomic_json(out/'run_manifest.json',manifest)
    # This ledger's reservation is the upstream accounting mechanism, NOT a
    # measured local price. Keep per-decision cost_usd null in the real runner.
    ledger=Ledger(out/'ledger.jsonl',cap_usd=.02*(len(tasks)+1))
    runner=Runner(adapter,ledger,out/'raw',default_reserve_usd=.02)
    records=runner.run_all(tasks,results_path=out/'predictions.jsonl')
    result=summarize(tasks,records,ledger_charged=ledger.charged)
    result['nd2_scope']=manifest
    scorable=[t for t in tasks if t.expected is not None and not t.provenance.get('exclude_reason')]
    scorable_ids={t.id for t in scorable}
    correct=sum(bool(r.get('correct')) for r in records if r['task_id'] in scorable_ids)
    result['nd2_all_planned_success_fraction']=correct/len(scorable) if scorable else None
    result['unattempted_count']=len(tasks)-len(records)
    atomic_json(out/'summary.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','tasks','out','upstream-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args(argv)
    print(json.dumps(run_public(a.bundle,a.tasks,a.out,a.upstream_commit,a.allow_smoke),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
