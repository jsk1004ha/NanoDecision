#!/usr/bin/env python3
"""Offline integration check only. It does not measure language understanding."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
from nanodecision2.initialization import initialize
from nanodecision2.data import prepare_cache
from nanodecision2.training import train,read_model
from nanodecision2.export import export_bundle
from nanodecision2.evaluation import evaluate_cache,calibrate_bundle
from nanodecision2 import DecisionEngine


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.out)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new/empty output directory')
    root.mkdir(parents=True,exist_ok=True);initialize(root/'init',smoke=True)
    sizes={'train':48,'validation':12,'temperature':12,'conformal':20,'test':12}
    with (root/'smoke.jsonl').open('w',encoding='utf-8') as f:
        for part,n in sizes.items():
            for i in range(n):
                row={'id':f'{part}-{i}','group_id':f'{part}-{i}','partition':part,
                     'state':f'Case {part} {i}. '+('Tests passed.' if i%2 else 'Tests failed.'),
                     'type':'noul','question':'Did the tests pass?','target':bool(i%2),'synthetic':True}
                f.write(json.dumps(row)+'\n')
    prepare_cache([root/'smoke.jsonl'],root/'cache',root/'init',128,128)
    cfg=Namespace(init=str(root/'init'),cache=str(root/'cache'),out=str(root/'run'),device='cpu',epochs=2,
        batch_size=4,accumulation=2,encoder_lr=.001,head_lr=.002,layer_decay=.9,weight_decay=.01,
        warmup_ratio=.05,brier_weight=.2,rps_weight=.2,eval_every=3,save_every=3,log_every=3,
        patience=20,seed=42,threads=1,bf16=False,gradient_checkpointing=False,resume=None,stop_after=0)
    training=train(cfg);checkpoint=root/'run'/'best'
    info=json.loads((checkpoint/'init.json').read_text())
    export_bundle(read_model(checkpoint),root/'bundle',checkpoint,True,True,128,128,info)
    for part in ('temperature','conformal'):
        evaluate_cache(root/'bundle',root/'cache',part,root/(part+'.jsonl'))
    cal=calibrate_bundle(root/'bundle',root/'temperature.jsonl',root/'conformal.jsonl')
    metrics=evaluate_cache(root/'bundle',root/'cache','test',root/'test.jsonl')
    result=DecisionEngine(root/'bundle').decide('Tests passed.',{'done':{'type':'noul','question':'Did the tests pass?'}})
    report={'purpose':'Synthetic integration smoke test ONLY; no real model quality evidence',
            'training':training,'calibration':cal,'metrics':metrics,'prediction':result}
    (root/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'report':str(root/'report.json'),'smoke_only':True},indent=2))

if __name__=='__main__':main()
