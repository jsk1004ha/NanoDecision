"""One command: public downloads -> exact 5M decisions -> train -> calibrated native bundle.

Normal runs require the user's CUDA GPU. --smoke is a clearly marked, fully
local integration test and NEVER stands in for real data or a trained product.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .auto_sources import atomic_json,sha_file


def default_recipe():
    return json.loads((Path(__file__).parent/'recipes'/'balanced_5m.json').read_text(encoding='utf-8'))


def smoke_recipe():
    r=default_recipe()
    r.update(name='offline_fixture_ONLY',train_rows=120,
        reserve_rows={'validation':20,'temperature':20,'conformal':20,'test':20},
        disk_min_gib=0,max_state_tokens=384,max_pair_tokens=256,sources=[],smoke_only=True)
    for split in ('train','test'):
        for lang in ('en','ko','de'):
            r['sources'].append(dict(id=f'fixture-{lang}-{split}',adapter='fixture',language=lang,split=split,policy='split_train',priority=0,required=True))
    r['training'].update(epochs=1,english_reference_epochs=1,batch_size=4,accumulation=2,
        encoder_lr=.001,head_lr=.003,eval_every=20,save_every=10,log_every=10,validation_limit=0,
        bf16=False,gradient_checkpointing=False,threads=2)
    r['evaluation'].update(device='cpu',batch_size=4,native_parity_samples=3)
    return r


def smoke_rows(recipe):
    """Fabricated fixtures, labelled as such. Not a dataset generation feature."""
    result={}
    for spec in recipe['sources']:
        rows=[]
        for i in range(1800 if spec['split']=='train' else 60):
            color='red' if i%2 else 'blue'
            row=dict(id=spec['id']+':'+str(i),group_id=spec['id']+':'+str(i),
                origin_id=spec['id']+':'+str(i),source='OFFLINE_SYNTHETIC_FIXTURE',source_split=spec['split'],
                language=spec['language'],state=f"{spec['id']} record {i}; marker {color}",
                type='choice',question='Which marker is recorded?',criteria={'r':'red','b':'blue'},
                target='r' if color=='red' else 'b',task='fixture_color',derived=False,
                label_origin='fabricated_fixture_NOT_REAL_DATA',auto_policy='split_train')
            rows.append(row)
        result[spec['id']]=rows
    return result


@contextmanager
def exclusive_workdir(path):
    path.mkdir(parents=True,exist_ok=True);f=(path/'.run.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e:raise RuntimeError('Another process is using this work directory') from e
        else:
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as e:raise RuntimeError('Another process is using this work directory') from e
        yield
    finally:f.close()  # OS releases the advisory lock, including on crashes.


def code_fingerprint():
    h=hashlib.sha256();root=Path(__file__).parent
    for p in sorted([*root.glob('*.py'), *root.glob('_native*.so'), *root.glob('_native*.pyd')]):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def _preflight(work,recipe,device):
    import torch
    for package in ('numpy','safetensors'):
        if importlib.util.find_spec(package) is None:raise RuntimeError(f'Missing {package}; run the bootstrap launcher')
    if not recipe.get('smoke_only'):
        for package in ('tokenizers','pyarrow','huggingface_hub','transformers'):
            if importlib.util.find_spec(package) is None:raise RuntimeError(f'Missing {package}; run bash run_all.sh')
        if device!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('The 5M run requires CUDA. No slow, silent CPU fallback. Check the NVIDIA driver and CUDA PyTorch wheel.')
        if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 is required by this recipe')
        # Include a real device allocation/matmul (not cuda.is_available alone).
        x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16);y=x@x
        if not torch.isfinite(y).all():raise RuntimeError('CUDA sanity check failed')
        torch.cuda.synchronize()
    from . import _native
    free=shutil.disk_usage(work).free
    if free<recipe['disk_min_gib']*(1<<30):raise RuntimeError(f'Insufficient free disk: {free/(1<<30):.1f} GiB; reserve {recipe["disk_min_gib"]} GiB for this recipe')
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for dep in ('nanodecision2','torch','numpy','safetensors','transformers','tokenizers','huggingface-hub','pyarrow'):
        try:versions[dep]=version(dep)
        except PackageNotFoundError:versions[dep]=None
    return {'installed_packages':versions,'python':sys.version,'torch':torch.__version__,'device':device,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'gpu_bytes':torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None,
            'disk_free_bytes':free,'smoke_only':recipe.get('smoke_only',False)}


def _initialize(work,recipe):
    from .initialization import initialize
    target=work/'init'
    if (target/'init.json').exists():return json.loads((target/'init.json').read_text())
    temp=work/'init.building'
    if temp.exists():shutil.rmtree(temp)
    revision=recipe.get('backbone_revision','main')
    if not recipe.get('smoke_only'):
        lock=work/'model_revision.lock.json'
        if lock.exists():revision=json.loads(lock.read_text())['revision']
        else:
            from huggingface_hub import HfApi
            from .auto_sources import retry
            revision=retry(lambda:HfApi().model_info(recipe.get('backbone','jhu-clsp/mmBERT-base'),revision=revision).sha)
            atomic_json(lock,{'backbone':recipe.get('backbone','jhu-clsp/mmBERT-base'),'revision':revision})
    r=initialize(temp,backbone=recipe.get('backbone','jhu-clsp/mmBERT-base'),
        revision=revision,smoke=recipe.get('smoke_only',False))
    os.replace(temp,target);return r


def _train(work,recipe,device,reference):
    from .cli import main as nd2
    t=recipe['training'];out=work/('english_reference' if reference else 'mixed_training')
    init=work/'init' if reference else work/'english_reference'/'best'
    report=out/'training_report.json'
    if report.exists() and json.loads(report.read_text()).get('completed'):return json.loads(report.read_text())
    if out.exists() and any(out.iterdir()) and not (out/'last.pt').exists():
        os.replace(out,out.with_name(out.name+'.uncheckpointed-'+str(time.time_ns())))
    args=['train','--init',str(init),'--cache',str(work/'cache'),'--out',str(out),'--device',device]
    mapping={'epochs':t['english_reference_epochs'] if reference else t['epochs'], 'batch-size':t['batch_size'],
        'accumulation':t['accumulation'],'encoder-lr':t['encoder_lr'],'head-lr':t['head_lr'],
        'eval-every':t['eval_every'],'save-every':t['save_every'],'log-every':t['log_every'],
        'validation-limit':t['validation_limit'],'patience':t['patience'],'threads':t['threads'],'seed':recipe['seed'],
        'selection-metric':'english_nll' if reference else 'macro_language_nll'}
    for k,v in mapping.items():args+=['--'+k,str(v)]
    if reference:args+=['--train-language','en']
    else:args+=['--guard-english','--guard-korean']
    if t['bf16']:args+=['--bf16']
    if t['gradient_checkpointing']:args+=['--gradient-checkpointing']
    if (out/'last.pt').exists():args+=['--resume',str(out/'last.pt')]
    nd2(args)
    report=json.loads(report.read_text())
    if not report.get('completed'):raise RuntimeError('Training checkpoint saved; rerun the same command to continue. It is not complete.')
    return report


def _export(work,which):
    from .training import read_model
    from .export import export_bundle
    source=work/('english_reference' if which=='reference' else 'mixed_training')/'best'
    out=work/'exports'/which
    if (out/'manifest.json').exists():return json.loads((out/'manifest.json').read_text())
    temp=out.with_name(out.name+'.building')
    if temp.exists():shutil.rmtree(temp)
    out.parent.mkdir(parents=True,exist_ok=True)
    info=json.loads((source/'init.json').read_text())
    r=export_bundle(read_model(source),temp,source,which=='int8',info.get('smoke_only',False),
        info['max_state_tokens'],info['max_pair_tokens'],info)
    os.replace(temp,out);return r


def worker(spec):
    work=Path(spec['work']);recipe=spec['recipe'];name=spec['name'];device=spec['device']
    if name=='preflight':r=_preflight(work,recipe,device)
    elif name=='initialize':r=_initialize(work,recipe)
    elif name=='data':
        from .autodata import build
        r=build(recipe,work,work/'init',smoke_rows(recipe) if recipe.get('smoke_only') else None)
    elif name=='english_reference':r=_train(work,recipe,device,True)
    elif name=='mixed_training':r=_train(work,recipe,device,False)
    elif name.startswith('export_'):r=_export(work,name.removeprefix('export_'))
    elif name.startswith('eval_'):
        from .auto_eval import evaluate
        _,which,split=name.split('_')
        limit=recipe['training']['validation_limit'] if split=='validation' else 0
        r=evaluate(work/'exports'/which,work/'cache',split,work/'reports'/f'{which}-{split}.jsonl',
            device=device,batch_size=recipe['evaluation']['batch_size'],parity_samples=recipe['evaluation']['native_parity_samples'],limit=limit,seed=recipe['seed'])
    elif name.startswith('calibrate_'):
        from .evaluation import calibrate_bundle
        which=name.removeprefix('calibrate_')
        r=calibrate_bundle(work/'exports'/which,work/'reports'/f'{which}-temperature.jsonl',
            work/'reports'/f'{which}-conformal.jsonl',language_calibration=True,
            min_bucket_groups=2 if recipe.get('smoke_only') else 200)
    else:raise ValueError('Unknown pipeline stage: '+name)
    atomic_json(work/'stage_results'/f'{name}.json',r)
    print(json.dumps({'stage':name,'status':'complete'}),flush=True)


class Pipeline:
    def __init__(self,work,recipe,device):
        self.work=Path(work);self.recipe=recipe;self.device=device
        self.state_path=self.work/'pipeline_state.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {'stages':{}}
        signature=hashlib.sha256(json.dumps({'recipe':recipe,'code':code_fingerprint(),'device':device},sort_keys=True).encode()).hexdigest()
        if self.state.get('signature',signature)!=signature:raise RuntimeError('Code/recipe/device changed. Resume with the original settings or a new --workdir.')
        self.state['signature']=signature
        for p in ('logs','stage_results','reports','stage_specs'): (self.work/p).mkdir(exist_ok=True)
        atomic_json(self.work/'recipe.lock.json',recipe)

    def artifacts(self,name):
        w=self.work
        if name=='initialize':return [w/'init'/x for x in ('model.safetensors','config.json','init.json')]
        if name=='data':
            manifest=json.loads((w/'cache'/'manifest.json').read_text())
            return [w/'cache'/'manifest.json']+[w/'cache'/(s+ext) for s,fs in manifest['files'].items() for ext in fs]
        if name in ('english_reference','mixed_training'):
            return [w/name/'best'/x for x in ('model.safetensors','config.json','init.json')]+[w/name/'training_report.json']
        if name.startswith('export_'):
            return [w/'exports'/name.removeprefix('export_')/x for x in ('model.nd2','manifest.json')]
        if name.startswith('calibrate_'):return [w/'exports'/name.removeprefix('calibrate_')/'calibration.json']
        if name.startswith('eval_'):
            _,which,split=name.split('_')
            return [w/'reports'/f'{which}-{split}.jsonl',w/'reports'/f'{which}-{split}.summary.json']
        return []

    def stage(self,name):
        prior=self.state['stages'].get(name,{})
        result_path=self.work/'stage_results'/f'{name}.json'
        if prior.get('status')=='complete' and result_path.exists() and sha_file(result_path)==prior.get('result_sha256'):
            for filename,expected in prior.get('artifacts',{}).items():
                path=Path(filename)
                if not path.exists() or sha_file(path)!=expected:
                    raise RuntimeError(f'Completed-stage artifact changed/missing: {path}. Refusing unsafe resume.')
            print(json.dumps({'stage':name,'status':'resumed_skip'}),flush=True);return
        self.state['current_stage']=name;self.state['status']='running'
        self.state['stages'][name]={'status':'running','started':time.time()};atomic_json(self.state_path,self.state)
        spec={'name':name,'work':str(self.work),'recipe':self.recipe,'device':self.device}
        spec_path=self.work/'stage_specs'/f'{name}.json';atomic_json(spec_path,spec)
        cmd=[sys.executable,'-u','-m','nanodecision2.autopipeline','_worker',str(spec_path)]
        environment={**os.environ,'HF_HOME':str(self.work/'downloads'/'hf_home'),'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
        print(json.dumps({'stage':name,'status':'starting'}),flush=True)
        started=time.monotonic()
        with (self.work/'logs'/f'{name}.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=environment)
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                rc=child.wait()
            except KeyboardInterrupt:
                # SIGINT is sent to the foreground process group by the terminal.
                # Give the training worker time to commit its current optimizer step.
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:child.terminate();child.wait()
                rc=130
        if rc or not result_path.exists():
            self.state['status']='failed';self.state['stages'][name].update(status='failed',returncode=rc)
            atomic_json(self.state_path,self.state)
            raise RuntimeError(f'{name} did not finish. See {self.work/"logs"/(name+".log")}. Rerun the same command to resume.')
        self.state['stages'][name].update(status='complete',seconds=time.monotonic()-started,result_sha256=sha_file(result_path),
                                        artifacts={str(p):sha_file(p) for p in self.artifacts(name)})
        atomic_json(self.state_path,self.state)

    def calibrated(self,which):
        for stage in ('export_'+which,'eval_'+which+'_temperature','eval_'+which+'_conformal','calibrate_'+which,'eval_'+which+'_validation'):
            self.stage(stage)

    def run(self,stop_after=None):
        for stage in ('preflight','initialize','data','english_reference','mixed_training'):
            self.stage(stage)
            if stage==stop_after:
                self.state['status']='stopped_after_requested_stage';atomic_json(self.state_path,self.state);return self.state
        self.calibrated('reference');self.calibrated('int8')
        from .train_views import bilingual_gate
        r=json.loads((self.work/'reports'/'reference-validation.summary.json').read_text())
        c=json.loads((self.work/'reports'/'int8-validation.summary.json').read_text())
        selected='int8';gate=bilingual_gate(c,r)
        if not gate:
            self.calibrated('fp32')
            c=json.loads((self.work/'reports'/'fp32-validation.summary.json').read_text())
            selected='fp32' if bilingual_gate(c,r) else 'reference'
        self.stage('eval_'+selected+'_test')
        candidate_info=json.loads((self.work/'mixed_training'/'best'/'init.json').read_text())
        improvement=selected!='reference' and candidate_info.get('selection_origin')=='mixed_candidate'
        status='mixed_candidate_passed_validation_gate' if improvement else 'reference_fallback_no_verified_mixed_improvement'
        final=self.work/'bundle';source=self.work/'exports'/selected
        if final.exists():
            m=json.loads((final/'manifest.json').read_text())
            if m['model_sha256']!=json.loads((source/'manifest.json').read_text())['model_sha256'] or any(not (final/p.name).is_file() or sha_file(final/p.name)!=sha_file(p) for p in source.iterdir() if p.is_file()):
                raise ValueError('Existing final bundle differs; refusing to overwrite')
        else:
            temp=final.with_name('bundle.building')
            if temp.exists():shutil.rmtree(temp)
            shutil.copytree(source,temp);os.replace(temp,final)
        data=json.loads((self.work/'data_report.json').read_text())
        test=json.loads((self.work/'reports'/f'{selected}-test.summary.json').read_text())
        report={'version':'0.3.1','status':'complete','quality_status':status,'selected_format':selected,
            'bundle':str(final),'smoke_only':self.recipe.get('smoke_only',False),
            'train_decisions':data['counts']['train'],'all_splits':data['counts'],
            'train_original_records':data['train_original_records'],'train_derived_decisions':data['train_derived_decisions'],
            'by_language':data['by_language'],'data_cache_bytes':data['disk_bytes'],
            'workdir_logical_bytes':sum(p.stat().st_size for p in self.work.rglob('*') if p.is_file() and not p.is_symlink()),
            'training_reports':{k:json.loads((self.work/k/'training_report.json').read_text()) for k in ('english_reference','mixed_training')},
            'english_guard_scope':'Observed accuracy and NLL on a fixed held-out validation set against this run\'s English-trained reference; Korean accuracy/NLL also must not regress. NOT a universal non-degradation guarantee or Julia/Laya result.',
            'evaluation':test,'stage_seconds':{k:v.get('seconds') for k,v in self.state['stages'].items()},
            'optional_components_not_run':['100M-token extra MLM','Julia and Laya external benchmark'],
            'next':'Use nanodecision2.DecisionEngine(bundle) for native CPU inference. Inspect quality_status before use.'}
        atomic_json(self.work/'FINAL_REPORT.json',report)
        self.state.update(status='complete',quality_status=status);atomic_json(self.state_path,self.state)
        return report


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='_worker':
        try:worker(json.loads(Path(argv[1]).read_text()));return
        except BaseException:
            traceback.print_exc();sys.exit(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',default='runs/auto5m');p.add_argument('--recipe')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after-stage',choices=('preflight','initialize','data','english_reference','mixed_training'))
    a=p.parse_args(argv)
    if a.smoke and a.recipe:p.error('--smoke uses its own fixture recipe')
    recipe=smoke_recipe() if a.smoke else json.loads(Path(a.recipe).read_text(encoding='utf-8')) if a.recipe else default_recipe()
    device='cpu' if a.smoke else a.device
    if a.smoke and a.workdir=='runs/auto5m':a.workdir='runs/offline_smoke031'
    work=Path(a.workdir).expanduser().resolve()
    if a.smoke:print('OFFLINE FIXTURE TEST ONLY: no downloaded data, no useful model.',flush=True)
    try:
        with exclusive_workdir(work):
            result=Pipeline(work,recipe,device).run(a.stop_after_stage)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except (ValueError,RuntimeError,OSError) as e:
        p.exit(2,'ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
