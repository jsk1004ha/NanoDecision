"""Padding-budgeted text training, R-Drop, per-task guards and torchrun DDP.

Run with ``nd2 train-scalable`` or ``torchrun ... -m
nanodecision2.scalable_training --ddp``. The existing trainer is unchanged.
No downloaded data, CUDA speedup, or language improvement is implied by a run.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from .auto_sources import atomic_json
from .autopipeline import exclusive_workdir
from .batching import cache_shapes, token_batches, distributed_rounds, update_groups
from .data import CacheDataset, collate
from .distributed_support import DistributedRun
from .loading import row_batches
from .losses import proper_loss
from .multilingual import reviewed_pair_units, bilingual_js
from .quality_gates import QualityReport, protected_slices, selection_key
from .runtime import sha256
from .train_views import training_view
from .training import read_model, save_inference_checkpoint, parameter_groups, atomic_save, INPUTS


def move_batch(batch, device, pin=False):
    if device.type == 'cuda' and pin:
        batch = {k: value.pin_memory() if not value.is_pinned() else value for k, value in batch.items()}
    return {k: v.to(device, non_blocking=pin and device.type == 'cuda') for k, v in batch.items()}


def symmetric_kl(first, second, mask):
    a = F.log_softmax(first.float().masked_fill(~mask, -1e4), -1)
    b = F.log_softmax(second.float().masked_fill(~mask, -1e4), -1)
    return .5 * ((a.exp() - b.exp()) * (a - b) * mask).sum(-1)


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size):
    model.eval(); report = QualityReport()
    for lo in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(lo, min(lo + batch_size, len(dataset)))]
        batch = move_batch(collate(rows, model.cfg.pad_token_id), device)
        z = model(**{k: batch[k] for k in INPUTS}).float().cpu().tolist()
        for row, logits in zip(rows, z):
            report.add(row, logits)
    return report.summary()


def train(args):
    for key in ('epochs','batch_size','accumulation','eval_every','eval_batch_size','save_every','threads'):
        if getattr(args, key) < 1:
            raise ValueError(key + ' must be positive')
    if not 0 <= args.dropout < 1 or min(args.rdrop,args.pair_consistency,args.max_tokens,args.stop_after) < 0:
        raise ValueError('Invalid loss/budget/dropout/stop configuration')
    if not all(math.isfinite(x) for x in (args.encoder_lr,args.head_lr,args.rdrop,args.pair_consistency,args.dropout)):
        raise ValueError('Non-finite training argument')
    if min(args.encoder_lr,args.head_lr) <= 0:
        raise ValueError('Learning rates must be positive')
    torch.set_num_threads(args.threads)
    run = DistributedRun(args.ddp, args.device)
    lock = []; dataset = validation = None
    try:
        random.seed(args.seed + run.rank); torch.manual_seed(args.seed + run.rank)
        if args.bf16 and (run.device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 requires a supported CUDA device')
        if args.fused and run.device.type != 'cuda':
            raise ValueError('Fused AdamW is an explicit CUDA-only option in this trainer')
        out = Path(args.out).resolve()
        def initialize_output():
            context = exclusive_workdir(out); context.__enter__(); lock.append(context)
            if not args.resume and any(p.name != '.run.lock' for p in out.iterdir()):
                raise FileExistsError('Use a fresh output or --resume')
        run.primary(initialize_output)
        dataset = CacheDataset(args.cache, 'train'); validation = CacheDataset(args.cache, 'validation')
        run.primary(lambda: (dataset.verify(), validation.verify()))
        if args.train_language:
            run.primary(lambda:len(training_view(dataset,args.train_language)))
            dataset=training_view(dataset,args.train_language)
        if not len(dataset) or not len(validation):
            raise ValueError('Nonempty train and validation required')
        info = json.loads((Path(args.init)/'init.json').read_text())
        if info.get('tokenizer_sha256') != dataset.manifest['tokenizer_sha256']:
            raise ValueError('Tokenizer/cache mismatch')
        if info.get('data_fingerprint') not in (None, dataset.manifest['fingerprint']):
            if not (info.get('group_assignment_sha256') == dataset.manifest.get('group_assignment_sha256') and
                    info.get('input_files_sha256') == dataset.manifest.get('input_files_sha256')):
                raise ValueError('Warm start exposure differs; audit and prepare compatible partitions')
        model = read_model(args.init).to(run.device)
        if max(dataset.manifest['max_state_tokens'], dataset.manifest['max_pair_tokens']) > model.cfg.max_position_embeddings:
            raise ValueError('Cache exceeds backbone position limit')
        model.encoder.gradient_checkpointing = args.gradient_checkpointing
        model.head.dropout.p = args.dropout
        run.primary(lambda: cache_shapes(dataset).shape[0])
        shapes = cache_shapes(dataset)
        units = reviewed_pair_units(dataset) if args.pair_consistency else None
        def batches(epoch):
            return token_batches(shapes, args.batch_size, args.max_tokens, args.seed, epoch, units)
        def rounds(epoch):
            return distributed_rounds(batches(epoch), run.world)
        epoch_rounds = [sum(1 for _ in rounds(ep)) for ep in range(args.epochs)]
        total_steps = sum(math.ceil(n/args.accumulation) for n in epoch_rounds)
        signature = {k: getattr(args,k) for k in ('epochs','batch_size','max_tokens','accumulation','encoder_lr',
            'head_lr','seed','bf16','fused','dropout','rdrop','pair_consistency','gradient_checkpointing','protect_language','eval_every','train_language')}
        training_code={name:sha256(Path(__file__).with_name(name)) for name in ('scalable_training.py','model.py','losses.py','batching.py','loading.py','multilingual.py','data.py','distributed_support.py','quality_gates.py','train_views.py')}
        signature.update(version=1, code_sha256=__import__('hashlib').sha256(json.dumps(training_code,sort_keys=True).encode()).hexdigest(), world_size=run.world, device_type=run.device.type,
                         data_fingerprint=dataset.manifest['fingerprint'], init_sha256=sha256(Path(args.init)/'model.safetensors'))
        optimizer = torch.optim.AdamW(parameter_groups(model,args.encoder_lr,args.head_lr,.9,.01),
                                     **({'fused': True} if args.fused else {}))
        wrapped = DistributedDataParallel(model, device_ids=[run.local_rank] if run.device.type == 'cuda' else None,
                                           broadcast_buffers=False) if run.world > 1 else model
        info = {**info, 'data_fingerprint': dataset.manifest['fingerprint'],
                'group_assignment_sha256':dataset.manifest['group_assignment_sha256'],
                'input_files_sha256':dataset.manifest['input_files_sha256'],
                'max_state_tokens':dataset.manifest['max_state_tokens'],'max_pair_tokens':dataset.manifest['max_pair_tokens'],
                'training_signature':signature,'trainer':'scalable-v1'}
        epoch = cursor = step = exposures = 0; best_key = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
            if checkpoint['signature'] != signature:
                raise ValueError('Resume signature changed; no implicit optimizer migration')
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            epoch, cursor, step, exposures = (checkpoint[k] for k in ('epoch','next_round','step','exposures'))
            best_key, reference = tuple(checkpoint['best_key']), checkpoint['reference']
            run.restore_rng(checkpoint['rng'])
        else:
            reference = run.primary(lambda: evaluate(model, validation, run.device, args.eval_batch_size))
            protected_slices(reference, reference, args.protect_language)
            best_key = selection_key(reference)
            run.primary(lambda: save_inference_checkpoint(model,out/'best',args.init,{**info,'selection_origin':'english_reference_fallback'}))
            run.primary(lambda: atomic_json(out/'reference.json',reference))
        started = time.monotonic(); local_rows = local_tokens = 0; interrupted = False
        def snapshot(next_epoch, next_round):
            rng = run.rng_states()
            run.primary(lambda: atomic_save({'signature':signature,'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'epoch':next_epoch,'next_round':next_round,'step':step,
                'exposures':exposures,'best_key':best_key,'reference':reference,'rng':rng},out/'last.pt'))
        for ep in range(epoch, args.epochs):
            first = cursor if ep == epoch else 0
            plan = [r[run.rank] for r in rounds(ep)]
            loader = iter(row_batches(dataset, (), args.batch_size, first,
                                      [ids or [0] for ids in plan], args.workers,args.prefetch,args.seed+ep))
            groups = update_groups(range(first, len(plan)), args.accumulation)
            model.train()
            for group in groups:
                optimizer.zero_grad(set_to_none=True); weight_sum = loss_sum = 0.; counted = 0
                for position, round_index in enumerate(group):
                    bi, rows = next(loader)
                    if bi != round_index:raise RuntimeError('Consumed loader cursor mismatch')
                    dummy = not plan[round_index]
                    batch = move_batch(collate(rows, model.cfg.pad_token_id, shuffle=True),run.device,args.pin_memory)
                    if dummy:batch['weight'].zero_()
                    with run.sync_context(wrapped, position == len(group)-1):
                        with torch.autocast(run.device.type,dtype=torch.bfloat16,enabled=args.bf16):
                            z = wrapped(**{k:batch[k] for k in INPUTS})
                            loss,_ = proper_loss(z,batch['target'],batch['option_mask'],batch['values'],batch['ordinal_mask'])
                            if args.rdrop:
                                other = wrapped(**{k:batch[k] for k in INPUTS})
                                other_loss,_ = proper_loss(other,batch['target'],batch['option_mask'],batch['values'],batch['ordinal_mask'])
                                loss = .5*(loss+other_loss) + args.rdrop*symmetric_kl(z,other,batch['option_mask'])
                            weighted = (loss*batch['weight']).sum()
                            if args.pair_consistency:
                                js,_ = bilingual_js(z,rows,batch['original_order'])
                                weighted = weighted + args.pair_consistency*js*batch['weight'].sum()
                        if run.sum(float(not torch.isfinite(weighted))) > 0:
                            raise RuntimeError('Non-finite loss on at least one rank')
                        weighted.backward()
                    weight_sum += float(batch['weight'].sum()); loss_sum += float(weighted.detach())
                    if not dummy:
                        counted += len(rows); local_rows += len(rows)
                        local_tokens += sum(len(r['state_ids']) + sum(map(len,r['pair_ids'])) for r in rows)
                denominator = run.scale_gradients(model,weight_sum)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                warmup = max(1,round(total_steps*.05)); upcoming=step+1
                multiplier=min(1.,upcoming/warmup) if upcoming<=warmup else .1+.45*(1+math.cos(math.pi*(upcoming-warmup)/max(1,total_steps-warmup)))
                for param_group in optimizer.param_groups:param_group['lr']=param_group['initial_lr']*multiplier
                optimizer.step(); step += 1; exposures += int(run.sum(counted))
                entry={'step':step,'epoch':ep+1,'global_rows_seen':exposures,'weighted_loss':run.sum(loss_sum)/denominator}
                end_epoch = group[-1]+1 == len(plan)
                if step % args.eval_every == 0 or end_epoch:
                    report=run.primary(lambda:evaluate(model,validation,run.device,args.eval_batch_size))
                    gate=protected_slices(report,reference,args.protect_language)
                    key=selection_key(report); entry.update(validation=report,protected_gate=gate)
                    if gate['passed'] and key < best_key:
                        best_key=key
                        run.primary(lambda:save_inference_checkpoint(model,out/'best',args.init,{**info,'selection_origin':'validation_best' if args.train_language else 'mixed_candidate'}))
                    model.train()
                def log_entry():
                    with (out/'history.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n')
                run.primary(log_entry)
                next_epoch,next_round=(ep+1,0) if end_epoch else (ep,group[-1]+1)
                stopping=bool(args.stop_after and step>=args.stop_after)
                if step%args.save_every==0 or end_epoch or stopping:snapshot(next_epoch,next_round)
                if stopping:interrupted=step<total_steps;break
            # Close workers before beginning validation/next epoch or early stop.
            if hasattr(loader,'close'):loader.close()
            if interrupted:break
        run.primary(lambda:save_inference_checkpoint(model,out/'final',args.init,info))
        rows=run.sum(local_rows);tokens=run.sum(local_tokens);elapsed=time.monotonic()-started
        report={'completed':not interrupted,'training_complete':not interrupted,'release_approved':False,'world_size':run.world,
                'device':str(run.device),'optimizer_steps':step,'planned_steps':total_steps,'global_rows_seen':exposures,
                'rows_this_invocation':rows,'nonpadding_tokens_this_invocation':tokens,'wall_seconds':elapsed,
                'rows_per_second':rows/max(elapsed,1e-9),'tokens_per_second':tokens/max(elapsed,1e-9),
                'best_selection_key':best_key,'protected_languages':args.protect_language,
                'cuda_peak_allocated':torch.cuda.max_memory_allocated() if run.device.type=='cuda' else None,
                'scope':'Training/validation only. Independent final evaluation and deployed calibration remain required.'}
        run.primary(lambda:atomic_json(out/'scalable_report.json',report))
        run.primary(lambda:atomic_json(out/'training_report.json',report))
        return report
    finally:
        if dataset is not None:dataset.close()
        if validation is not None:validation.close()
        if lock:lock[0].__exit__(None,None,None)
        run.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('init','cache','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--ddp',action='store_true')
    for name,default in [('epochs',1),('batch-size',4),('max-tokens',0),('accumulation',16),('eval-every',500),
                         ('eval-batch-size',2),('save-every',100),('threads',2),('workers',0),('prefetch',2),('seed',42),('stop-after',0)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('encoder-lr',1e-5),('head-lr',5e-5),('rdrop',0.),('pair-consistency',0.),('dropout',.1)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--protect-language',action='append',default=None,choices=('en','ko','mixed'))
    p.add_argument('--no-language-guard',action='store_true',help='Explicit research-only ablation; never a release approval')
    for name in ('bf16','fused','pin-memory','gradient-checkpointing'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--resume');p.add_argument('--train-language',choices=['en','ko','mixed'])
    return p


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.no_language_guard and a.protect_language:p.error('Choose guards or explicit no-guard ablation, not both')
    a.protect_language=[] if a.no_language_guard else (a.protect_language or ['en'])
    report=train(a)
    if int(__import__('os').environ.get('RANK','0'))==0:print(json.dumps(report,indent=2))

if __name__=='__main__':main()
