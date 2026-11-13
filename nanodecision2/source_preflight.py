"""Bounded parallel source download/probing before encoder initialization.

Resolution and lock-file publication happen in the coordinator. Workers probe
immutable resolved specs; duplicate first-shard downloads share a keyed lock.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
from itertools import islice
import threading
from .auto_sources import PublicReader, raw_rows, convert, atomic_json
from .source_contracts import SOURCE_CONTRACT_VERSION
from .schema import normalize_question, gold_target


def preflight(recipe, workdir, sample_records=3, workers=1, prefetch=2):
    if type(sample_records) is not int or not 1 <= sample_records <= 1000:
        raise ValueError('sample_records must be 1..1000')
    if type(workers) is not int or not 1<=workers<=16 or not 1<=prefetch<=8:
        raise ValueError('Invalid source worker/prefetch limits')
    root=Path(workdir);reader=PublicReader(root/'downloads')
    report={'source_contract_version':SOURCE_CONTRACT_VERSION,'sources':[],
            'scope':'First-shard sample validation, not a full-corpus audit',
            'workers':workers,'max_in_flight':workers*prefetch}
    specs=[s for s in recipe['sources'] if s['adapter']!='nli_entails' or recipe.get('derived_nli_entailment',False)]
    jobs=[];locks={};required_failures=[]
    # Resolve sequentially so worker scheduling never changes sources.lock.json.
    for spec in specs:
        result={'id':spec['id'],'required':spec.get('required',True)}
        try:
            resolved=reader.resolve(spec);result['resolved']=resolved
            if 'url' in resolved:key=('direct',tuple(resolved['files']))
            else:key=(resolved['repo'],resolved['data_revision'],resolved['files'][0])
            jobs.append((spec,result,locks.setdefault(key,threading.Lock())))
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__,error=str(exc));jobs.append((None,result,None))
    def probe(job):
        spec,result,lock=job
        if spec is None:return result
        try:
            iterator=reader.shards(spec)
            try:
                with lock:shard=next(iterator,None)
            finally:iterator.close()
            if shard is None:raise ValueError('Source has no shards')
            shard_id,path,provenance=shard;reading=dict(spec)
            if provenance.get('file','').endswith('.parquet'):reading['_parquet']=True
            seen=decisions=0;rows=raw_rows(path,reading)
            try:
                for index,raw,names in islice(rows,sample_records):
                    seen+=1
                    for decision in convert(spec,raw,f'{shard_id}:{index}',names):
                        gold_target(decision,normalize_question(decision));decisions+=1
            finally:rows.close()
            if not seen:raise ValueError('No records in source sample')
            result.update(status='ok',sampled_records=seen,converted_decisions=decisions,shard=provenance)
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        return result
    def publish(result):
        report['sources'].append(result)
        if result['status']=='failed' and result['required']:required_failures.append(result['id'])
        report['required_failures']=list(required_failures);atomic_json(root/'source_preflight.json',report)
    if workers==1:
        for job in jobs:publish(probe(job))
    else:
        queue=deque();iterator=iter(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(workers*prefetch):
                job=next(iterator,None)
                if job is None:break
                queue.append(pool.submit(probe,job))
            while queue:
                publish(queue.popleft().result())
                job=next(iterator,None)
                if job is not None:queue.append(pool.submit(probe,job))
    if required_failures:
        raise RuntimeError('Required source preflight failed: '+', '.join(required_failures)+
                           '. See source_preflight.json; model download has not started.')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','workdir'):p.add_argument('--'+name,required=True)
    p.add_argument('--samples',type=int,default=3);p.add_argument('--workers',type=int,default=2);p.add_argument('--prefetch',type=int,default=2)
    a=p.parse_args(argv)
    print(json.dumps(preflight(json.loads(Path(a.recipe).read_text()),a.workdir,a.samples,a.workers,a.prefetch),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
