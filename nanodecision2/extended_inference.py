"""Explicit prediction/calibration for the optional PyTorch research backend."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from .extended_workflow import ExtendedEngine,RowIndex,encoded_target
from .probability import fit_temperature
from .runtime import sha256
from .auto_sources import atomic_json


@torch.inference_mode()
def calibrate(bundle,source,device='cpu',allow_smoke=False):
    engine=ExtendedEngine(bundle,device,allow_smoke)
    if sha256(source)!=engine.manifest.get('provenance',{}).get('training_data_sha256'):
        raise ValueError('Use the originally group-audited source with its held-out temperature partition')
    records=[]
    with tempfile.TemporaryDirectory() as temp:
        index=RowIndex(source,Path(temp)/'audit.sqlite')
        try:
            for i in index.ids('temperature'):
                row=index[i];spec,target=encoded_target(row)
                z=engine.model(row['state'],row['question_spec'],engine.codec).float().cpu().tolist()
                records.append({'logits':z,'target_probs':target})
        finally:index.close()
    if not records:raise ValueError('No independent temperature rows')
    fitted=fit_temperature(records)
    fitted.update(model_sha256=engine.manifest['model_sha256'],architecture_id=engine.manifest['architecture_id'],
                  data_sha256=sha256(source),method='temperature_only',
                  scope='Independent temperature partition in the original audited source; no test fitting.')
    atomic_json(Path(bundle)/'calibration.json',fitted);return fitted


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('predict','calibrate'):
        a=sub.add_parser(command);a.add_argument('--bundle',required=True);a.add_argument('--input',required=True)
        a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args(argv)
    if a.command=='calibrate':result=calibrate(a.bundle,a.input,a.device,a.allow_smoke)
    else:
        engine=ExtendedEngine(a.bundle,a.device,a.allow_smoke);request=json.loads(Path(a.input).read_text(encoding='utf-8'))
        result=engine.decide(request['state'],request['questions'],request.get('language'))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
