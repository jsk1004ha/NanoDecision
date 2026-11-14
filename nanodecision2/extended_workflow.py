"""Train/evaluate optional numeric, sensor and token-interaction research modules.

This opt-in backend requires PyTorch. It never labels its safetensors checkpoint
as a legacy native bundle, and never imports a target field into model inputs.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import time
import torch
from .auto_sources import atomic_json
from .config import ModelConfig
from .data import collate
from .decision_contract import calibration_language,decision_result
from .extended_model import ARCHITECTURE_ID,ExtensionConfig,ExtendedDecisionModel,auxiliary_terms
from .losses import proper_loss
from .multilingual import served_temperature
from .probability import softmax
from .quality_gates import QualityReport,protected_slices,selection_key
from .runtime import sha256
from .schema import TextCodec,normalize_question,gold_target,normalized_state_text
from .training import read_model


class RowIndex:
    """Disk-backed source offsets and group/state leakage audit, not an in-RAM corpus."""
    def __init__(self,path,index_path):
        self.path=Path(path);self.file=self.path.open('rb');self.db=sqlite3.connect(index_path)
        self.db.executescript('CREATE TABLE items(id INTEGER PRIMARY KEY,offset INTEGER,part TEXT);CREATE TABLE groups(g TEXT PRIMARY KEY,part TEXT);CREATE TABLE states(s TEXT PRIMARY KEY,part TEXT);CREATE TABLE ids(i TEXT PRIMARY KEY);')
        offset=0;counts={}
        for line in self.file:
            if line.strip():
                row=json.loads(line);part=row.get('partition');group=row.get('group_id');identifier=row.get('id')
                if part not in ('train','validation','temperature','test') or not isinstance(group,str) or not group or not isinstance(identifier,str) or not identifier:
                    raise ValueError('Every row needs unique id, original group_id, and explicit partition')
                # Native schemas are top-level inputs; supervision is isolated.
                question=row.get('question_spec')
                if not isinstance(question,dict):raise ValueError('question_spec object required')
                target=row.get('supervision',{})
                gold_target(target,normalize_question(question))
                for table,key in [('groups',group),('states',__import__('hashlib').sha256(normalized_state_text(row['state']).encode()).hexdigest())]:
                    old=self.db.execute(f'SELECT part FROM {table} WHERE '+('g' if table=='groups' else 's')+'=?',(key,)).fetchone()
                    if old and old[0]!=part:raise ValueError('Original group or exact input crosses partitions')
                    self.db.execute(f'INSERT OR IGNORE INTO {table} VALUES (?,?)',(key,part))
                self.db.execute('INSERT INTO ids VALUES (?)',(identifier,))
                self.db.execute('INSERT INTO items(offset,part) VALUES (?,?)',(offset,part));counts[part]=counts.get(part,0)+1
            offset+=len(line)
        self.db.commit();self.counts=counts
    def ids(self,part):return [v[0] for v in self.db.execute('SELECT id FROM items WHERE part=? ORDER BY id',(part,))]
    def __getitem__(self,index):
        offset=self.db.execute('SELECT offset FROM items WHERE id=?',(int(index),)).fetchone()
        if offset is None:raise IndexError(index)
        self.file.seek(offset[0]);return json.loads(self.file.readline())
    def close(self):self.file.close();self.db.close()


def encoded_target(row):
    spec=normalize_question(row['question_spec']);target=gold_target(row['supervision'],spec)
    return spec,target


def loss_for(z,spec,target):
    device=z.device
    y=torch.tensor([target],device=device);mask=torch.ones_like(y,dtype=torch.bool)
    values=torch.tensor([spec['values'] or [0.]*len(target)],device=device)
    loss,_=proper_loss(z[None],y,mask,values,torch.tensor([spec['type']=='score'],device=device))
    return loss[0]


@torch.inference_mode()
def evaluate(model,index,ids,codec):
    model.eval();report=QualityReport()
    for i in ids:
        row=index[i];spec,target=encoded_target(row)
        z=model(row['state'],row['question_spec'],codec).float().cpu().tolist()
        report.add({'type':spec['type'],'option_ids':[o['id'] for o in spec['options']],
            'target_probs':target,'values':spec['values'],'task':row.get('task','unspecified'),
            'language':calibration_language(row['question_spec'],row.get('language'),row.get('language_metadata'))},z)
    return report.summary()


def save_checkpoint(model,out,source,info):
    from safetensors.torch import save_file
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tmp=out/'model.safetensors.partial'
    save_file({k:v.detach().float().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp));tmp.replace(out/'model.safetensors')
    atomic_json(out/'config.json',model.cfg.to_dict());atomic_json(out/'extension.json',asdict(model.extension))
    smoke=info.get('smoke_only',False)
    if not smoke:shutil.copyfile(Path(source)/'tokenizer.json',out/'tokenizer.json')
    atomic_json(out/'manifest.json',{'architecture_id':ARCHITECTURE_ID,'backend':'pytorch_research',
        'model_sha256':sha256(out/'model.safetensors'),'config_sha256':sha256(out/'config.json'),
        'extension_sha256':sha256(out/'extension.json'),'tokenizer_sha256':'smoke_byte_v1' if smoke else sha256(out/'tokenizer.json'),
        'smoke_only':smoke,'supported_modalities':['text']+(['records','table'] if model.extension.structured else [])+(['timeseries'] if model.extension.sensors else []),
        'abstention_default':False,'release_approved':False,'provenance':info,
        'context_note':'All-block mode resets positions per block; not direct full-context attention.'})


class ExtendedEngine:
    def __init__(self,bundle,device='cpu',allow_smoke=False):
        from safetensors.torch import load_file
        self.root=Path(bundle);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('architecture_id')!=ARCHITECTURE_ID:raise ValueError('Wrong research architecture')
        self.smoke=self.manifest.get('smoke_only',False)
        if self.smoke and not allow_smoke:raise ValueError('Test-only checkpoint; explicit allow_smoke required')
        for file,key in [('model.safetensors','model_sha256'),('config.json','config_sha256'),('extension.json','extension_sha256')]:
            if sha256(self.root/file)!=self.manifest[key]:raise ValueError('Research checkpoint hash mismatch')
        if not self.smoke and sha256(self.root/'tokenizer.json')!=self.manifest['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
        self.model=ExtendedDecisionModel(ModelConfig(**json.loads((self.root/'config.json').read_text())),ExtensionConfig(**json.loads((self.root/'extension.json').read_text())))
        self.model.load_state_dict(load_file(str(self.root/'model.safetensors')),strict=True);self.model.to(device).eval()
        self.codec=TextCodec(self.root/'tokenizer.json',smoke=self.smoke);self.calibration=None
        if (self.root/'calibration.json').exists():
            self.calibration=json.loads((self.root/'calibration.json').read_text())
            if self.calibration['model_sha256']!=self.manifest['model_sha256']:raise ValueError('Calibration checkpoint mismatch')
            softmax([0.,0.],self.calibration['temperature'])
    @torch.inference_mode()
    def decide(self,state,questions,language=None):
        if not isinstance(questions,dict) or not 1<=len(questions)<=64:raise ValueError('Expected 1..64 named questions')
        result={}
        for name,question in questions.items():
            spec=normalize_question(question);z,_,context=self.model(state,question,self.codec,return_aux=True)
            z=z.float().cpu().tolist();lang=calibration_language(question,language)
            temp=served_temperature(self.calibration,lang,spec['type'],len(z))
            a=decision_result(spec,softmax(z,temp));a.update(logits=z,temperature_used=temp,calibrated=self.calibration is not None,
                context=context,backend='pytorch_research',smoke_only=self.smoke)
            if self.smoke:a['status']='test_only'
            result[name]=a
        return {'answers':result,'model_sha256':self.manifest['model_sha256'],'architecture_id':ARCHITECTURE_ID}


def train(args):
    if min(args.epochs,args.accumulation,args.threads)<=0 or not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.aux_weight) or args.aux_weight<0:raise ValueError('Invalid training settings')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh research output (no implicit resume)')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    ext=ExtensionConfig(structured=args.structured,sensors=args.sensors,token_interaction=args.token_interaction,
        evidence=args.evidence,relation_classes=args.relation_classes,block_tokens=args.block_tokens,
        max_fused_tokens=args.max_context,max_pair_tokens=args.max_pair,metadata_tokens=args.metadata_tokens,patch_size=args.patch_size)
    source=Path(args.init);base=read_model(source);info=json.loads((source/'init.json').read_text())
    model=ExtendedDecisionModel(base.cfg,ext);model.core.load_state_dict(base.state_dict());del base
    model.to(args.device);model.core.encoder.gradient_checkpointing=args.gradient_checkpointing
    codec=TextCodec(source/'tokenizer.json',smoke=info.get('smoke_only',False));index=RowIndex(args.input,out/'audit.sqlite')
    try:
        train_ids=index.ids('train');val_ids=index.ids('validation')
        if not train_ids or not val_ids:raise ValueError('Explicit train and validation data required')
        reference=evaluate(model,index,val_ids,codec);guards=args.protect_language or ['en']
        if args.no_language_guard:guards=[]
        protected_slices(reference,reference,guards);best=selection_key(reference)
        info.update(training_data_sha256=sha256(args.input),initial_weights_sha256=sha256(source/'model.safetensors'),reference=reference)
        save_checkpoint(model,out/'best',source,info);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        history=[];start=time.monotonic();rows_seen=0
        for epoch in range(args.epochs):
            order=list(train_ids);random.Random(args.seed+epoch).shuffle(order);model.train()
            for lo in range(0,len(order),args.accumulation):
                ids=order[lo:lo+args.accumulation];optimizer.zero_grad(set_to_none=True)
                # Supervision-specific denominators, not zeros for missing labels.
                rows=[index[i] for i in ids];aux_count={key:sum(field in r['supervision'] for r in rows) for key,field in [('start','evidence_token_span'),('end','evidence_token_span'),('relation','relation_id')]}
                for row in rows:
                    spec,target=encoded_target(row)
                    if 'evidence_token_span' in row['supervision'] and isinstance(row['state'],dict) and row['state'].get('modality') not in (None,'text'):
                        raise ValueError('Text evidence spans cannot supervise numeric patches')
                    z,aux,_=model(row['state'],row['question_spec'],codec,return_aux=True)
                    loss=loss_for(z,spec,target)/len(rows)
                    terms=auxiliary_terms(aux,row['supervision'])
                    for key,extra in terms.items():loss=loss+args.aux_weight*extra/max(1,aux_count[key])
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite research loss')
                    loss.backward();rows_seen+=1
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            report=evaluate(model,index,val_ids,codec);gate=protected_slices(report,reference,guards)
            if gate['passed'] and selection_key(report)<best:
                best=selection_key(report);save_checkpoint(model,out/'best',source,info)
            history.append({'epoch':epoch+1,'validation':report,'protected_gate':gate})
            atomic_json(out/'history.json',history)
        save_checkpoint(model,out/'final',source,info)
        report={'training_complete':True,'release_approved':False,'rows_seen':rows_seen,'wall_seconds':time.monotonic()-start,
                'architecture_id':ARCHITECTURE_ID,'backend':'pytorch_research','history':history,
                'native_export_supported':False,'scope':'Optional new graph. Real-task quality, deployed calibration and native implementation are separate validations.'}
        atomic_json(out/'extended_report.json',report);return report
    finally:index.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','input','out'):p.add_argument('--'+name,required=True)
    for name in ('structured','sensors','token-interaction','evidence','gradient-checkpointing','no-language-guard'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--protect-language',action='append',choices=['en','ko','mixed'])
    for name,default in [('epochs',1),('accumulation',4),('threads',2),('seed',42),('relation-classes',0),('block-tokens',2048),('max-context',32768),('max-pair',512),('metadata-tokens',256),('patch-size',16)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--aux-weight',type=float,default=.1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');return p

def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.no_language_guard and a.protect_language:p.error('Specify guards or no-guard, not both')
    print(json.dumps(train(a),indent=2))
if __name__=='__main__':main()
