"""Measure training micro-batches on an existing cache; never truncate to fit.

Probe executes forward/backward/AdamW at lr=0, including optimizer-state memory.
It does not alter checkpoint files or establish model quality.
"""
import argparse
import json
import time
import torch
from .auto_sources import atomic_json
from .data import CacheDataset,collate
from .training import read_model,INPUTS
from .scalable_training import move_batch
from .losses import proper_loss


def probe(init,cache,device='cuda',candidates=(1,2,4,8,16),steps=3,bf16=False,checkpointing=False):
    if steps<1 or any(type(n) is not int or not 1<=n<=256 for n in candidates):raise ValueError('Invalid probe budget')
    dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if bf16 and (dev.type!='cuda' or not torch.cuda.is_bf16_supported()):raise ValueError('BF16 unsupported')
    ds=CacheDataset(cache,'train');ds.verify()
    if not len(ds):raise ValueError('Empty training cache')
    # Inspect a bounded deterministic sample and choose its expensive rows.
    ids=sorted(set(round(i*(len(ds)-1)/max(1,min(len(ds),256)-1)) for i in range(min(len(ds),256))))
    sample=[ds[i] for i in ids];sample.sort(key=lambda r:len(r['state_ids'])+len(r['pair_ids'])*max(map(len,r['pair_ids'])),reverse=True)
    results=[]
    for size in candidates:
        model=opt=batch=None
        try:
            model=read_model(init).to(dev).train();model.encoder.gradient_checkpointing=checkpointing
            opt=torch.optim.AdamW(model.parameters(),lr=0.)
            rows=[sample[i%len(sample)] for i in range(size)]
            batch=move_batch(collate(rows,model.cfg.pad_token_id),dev,dev.type=='cuda')
            if dev.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            elapsed=[]
            for iteration in range(steps+1):
                start=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.autocast(dev.type,dtype=torch.bfloat16,enabled=bf16):
                    z=model(**{k:batch[k] for k in INPUTS})
                    loss,_=proper_loss(z,batch['target'],batch['option_mask'],batch['values'],batch['ordinal_mask'])
                loss.mean().backward();opt.step()
                if dev.type=='cuda':torch.cuda.synchronize()
                if iteration:elapsed.append(time.perf_counter()-start)
            results.append({'batch_size':size,'status':'ok','seconds_per_microbatch':sum(elapsed)/len(elapsed),
                'nonpadding_tokens':sum(len(r['state_ids'])+sum(map(len,r['pair_ids'])) for r in rows),
                'peak_allocated':torch.cuda.max_memory_allocated() if dev.type=='cuda' else None,
                'peak_reserved':torch.cuda.max_memory_reserved() if dev.type=='cuda' else None})
        except torch.OutOfMemoryError:
            results.append({'batch_size':size,'status':'oom'})
        finally:
            del model,opt,batch
            if dev.type=='cuda':torch.cuda.empty_cache()
    ds.close()
    return {'device':str(dev),'bf16':bf16,'checkpointing':checkpointing,'results':results,
            'scope':'Bounded sampled workload, not every possible input or full training throughput. No automatic production batch change.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','cache','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--candidates',type=int,nargs='+',default=[1,2,4,8,16]);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--bf16',action='store_true');p.add_argument('--gradient-checkpointing',action='store_true')
    a=p.parse_args(argv)
    result=probe(a.init,a.cache,a.device,a.candidates,a.steps,a.bf16,a.gradient_checkpointing)
    atomic_json(__import__('pathlib').Path(a.out),result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
