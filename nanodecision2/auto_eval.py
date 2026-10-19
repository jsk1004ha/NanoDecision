"""Batched FP32 evaluation of the exact weights stored in a native bundle.

On the user's GPU this avoids running thousands of reference C++ CPU forwards.
A sampled C++/PyTorch parity check is mandatory. Backend and numerical scope are
reported; GPU results are NOT mislabelled as a C++ performance benchmark.
"""
from pathlib import Path
import json
import struct
import time
import numpy as np
import torch
from .config import ModelConfig
from .model import DecisionModel
from .runtime import DecisionEngine,sha256
from .data import CacheDataset,collate
from .training import INPUTS
from .train_views import validation_view
from .probability import softmax,EvaluationAccumulator
from .multilingual import language_key,served_temperature
from .auto_sources import atomic_json


def load_bundle_weights(bundle):
    root=Path(bundle);manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/'model.nd2')!=manifest['model_sha256']:raise ValueError('Native weights changed')
    cfg=ModelConfig(**manifest['config']);model=DecisionModel(cfg);expected=model.state_dict();loaded={}
    with (root/'model.nd2').open('rb') as f:
        if f.read(8)!=b'NDC2BIN\0' or struct.unpack('<I',f.read(4))[0]!=1:raise ValueError('Bad native header')
        ints=struct.unpack('<11I',f.read(44));floats=struct.unpack('<3f',f.read(12));n=struct.unpack('<I',f.read(4))[0]
        keys=['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','global_attn_every_n_layers',
              'local_attention','max_position_embeddings','decision_dim','decision_hops','pad_token_id']
        if any(getattr(cfg,k)!=v for k,v in zip(keys,ints)):raise ValueError('Native header/config mismatch')
        if n!=len(expected):raise ValueError('Wrong tensor count')
        for _ in range(n):
            nl=struct.unpack('<I',f.read(4))[0]
            if not 1<=nl<=1024:raise ValueError('Invalid tensor name')
            name=f.read(nl).decode();nd,rows,cols,quant,nbytes=struct.unpack('<4IQ',f.read(24))
            if name in loaded or name not in expected:raise ValueError('Unexpected native tensor')
            shape=(rows,cols) if nd==2 else (rows,)
            if tuple(expected[name].shape)!=shape:raise ValueError('Native tensor shape mismatch')
            needed=rows*4+rows*cols if quant else rows*cols*4
            if nbytes!=needed:raise ValueError('Native payload length mismatch')
            f.seek((-f.tell())%64,1)
            payload=f.read(nbytes)
            if len(payload)!=nbytes:raise ValueError('Truncated native model')
            if quant:
                scales=np.frombuffer(payload,dtype='<f4',count=rows)
                q=np.frombuffer(payload,dtype='i1',offset=rows*4).reshape(rows,cols)
                a=q.astype(np.float32)*scales[:,None]
            else:a=np.frombuffer(payload,dtype='<f4').copy().reshape(shape)
            if not np.isfinite(a).all():raise ValueError('Non-finite native tensor')
            loaded[name]=torch.from_numpy(a)
        if f.read(1):raise ValueError('Unexpected trailing model bytes')
    model.load_state_dict(loaded,strict=True);return model.eval(),manifest


@torch.inference_mode()
def evaluate(bundle,cache,split,out,device='cpu',batch_size=2,parity_samples=8,limit=0,seed=20260929):
    torch.set_num_threads(4)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('GPU evaluation requested but CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    model,manifest=load_bundle_weights(bundle);model.to(device)
    ds=CacheDataset(cache,split);ds.verify()
    if manifest['provenance'].get('data_fingerprint')!=ds.manifest['fingerprint']:raise ValueError('Wrong cache for bundle')
    if limit:ds=validation_view(ds,limit,seed)
    if not len(ds):raise ValueError('Empty evaluation split')
    engine=DecisionEngine(bundle,require_calibration=False)
    # Sample across the split; don't only check an easy English first row.
    indices=sorted(set(int(x) for x in np.linspace(0,len(ds)-1,min(parity_samples,len(ds)))))
    worst=0.0
    for i in indices:
        row=ds[i];batch={k:v.to(device) for k,v in collate([row],model.cfg.pad_token_id).items()}
        ours=model(**{k:batch[k] for k in INPUTS})[0,:len(row['pair_ids'])].float().cpu().numpy()
        context=engine.native.prepare(engine.handle,row['state_ids'])
        native=np.asarray(engine.native.score(engine.handle,context,row['pair_ids']))
        worst=max(worst,float(np.abs(native-ours).max()))
        if not np.allclose(native,ours,atol=2e-3,rtol=2e-3):raise RuntimeError(f'Native/GPU FP32 parity failed: {worst}. Do not deploy.')
    acc=EvaluationAccumulator(engine.calibration.get('qhat') if engine.calibration else None);by_lang={}
    temp=out.with_name(out.name+'.partial');started=time.monotonic()
    with temp.open('w',encoding='utf-8') as f:
        for start in range(0,len(ds),batch_size):
            rows=[ds[i] for i in range(start,min(start+batch_size,len(ds)))]
            batch={k:v.to(device) for k,v in collate(rows,model.cfg.pad_token_id).items()}
            logits=model(**{k:batch[k] for k in INPUTS}).float().cpu().tolist()
            for row,z in zip(rows,logits):
                z=z[:len(row['option_ids'])]
                t=served_temperature(engine.calibration,row.get('language'),row['type'],len(z));p=softmax(z,t)
                result={k:row[k] for k in ('id','group_id','type','split','option_ids','target_probs','values')}
                result.update(language=row.get('language'),task=row['task'],label_origin=row.get('label_origin'),
                    logits=z,probabilities=p,temperature_used=t,model_sha256=manifest['model_sha256'],
                    data_fingerprint=ds.manifest['fingerprint'],backend='pytorch_fp32_exact_native_weights')
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
                acc.add(p,row['target_probs'],row['values'],row['group_id'])
                la=by_lang.setdefault(language_key(row.get('language')),EvaluationAccumulator())
                la.add(p,row['target_probs'],row['values'],row['group_id'])
            if start and start%1000<batch_size:print(json.dumps({'stage':'evaluation','split':split,'rows':start,'device':device}),flush=True)
    temp.replace(out)
    report={**acc.summary(),'by_language':{k:v.summary() for k,v in by_lang.items()},'model_sha256':manifest['model_sha256'],
            'split':split,'backend':'pytorch_fp32_exact_native_weights','device':device,'wall_seconds':time.monotonic()-started,
            'native_parity':{'samples':len(indices),'max_abs_logit_error':worst,'scope':'Sampled math check; not all-input guarantee or C++ throughput benchmark.'},
            'smoke_only':manifest.get('smoke_only',False)}
    atomic_json(out.with_suffix('.summary.json'),report);ds.close();return report
