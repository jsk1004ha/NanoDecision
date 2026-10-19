"""Deterministic, single-device, disk-cached training; no teacher/API required."""
import json
import math
from pathlib import Path
import random
import shutil
import time
import signal
import numpy as np
import torch
from .train_views import training_view,validation_view,DatasetView,english_gate,korean_gate
from .config import ModelConfig
from .model import DecisionModel
from .data import CacheDataset,collate
from .losses import proper_loss
from .probability import metrics,softmax
from .multilingual import language_key,validate_language_weights,reviewed_pair_units,paired_batches,bilingual_js

INPUTS=("state_ids","state_mask","pair_ids","pair_mask","option_mask","state_index")


def read_model(path):
    from safetensors.torch import load_file
    root=Path(path);cfg=ModelConfig(**json.loads((root/"config.json").read_text()))
    model=DecisionModel(cfg)
    model.load_state_dict(load_file(str(root/"model.safetensors")),strict=True)
    return model


def atomic_save(payload,path):
    temp=Path(str(path)+".tmp");torch.save(payload,temp);temp.replace(path)


def save_inference_checkpoint(model,out,source,info):
    from safetensors.torch import save_file
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tensors={k:v.detach().float().cpu().contiguous() for k,v in model.state_dict().items()}
    temp=out/"model.safetensors.tmp";save_file(tensors,str(temp));temp.replace(out/"model.safetensors")
    (out/"config.json").write_text(json.dumps(model.cfg.to_dict(),indent=2))
    (out/"init.json").write_text(json.dumps(info,indent=2))
    src=Path(source)/"tokenizer.json"
    if src.exists() and src.resolve()!=(out/"tokenizer.json").resolve():shutil.copyfile(src,out/"tokenizer.json")


def parameter_groups(model,encoder_lr,head_lr,layer_decay,weight_decay):
    grouped={}
    for name,p in model.named_parameters():
        if name.startswith("head."):lr=head_lr
        elif name.startswith("encoder.layers."):
            i=int(name.split(".")[2]);lr=encoder_lr*layer_decay**(model.cfg.num_hidden_layers-1-i)
        elif name.startswith("encoder.final_norm."):lr=encoder_lr
        else:lr=encoder_lr*layer_decay**model.cfg.num_hidden_layers
        decay=weight_decay if p.ndim>=2 else 0.0
        grouped.setdefault((lr,decay),[]).append(p)
    return [{"params":ps,"lr":lr,"initial_lr":lr,"weight_decay":decay} for (lr,decay),ps in grouped.items()]


@torch.inference_mode()
def validate(model,dataset,device,batch_size):
    model.eval();totals={};count=0;hard=correct=0;by_type={};by_language={}
    for start in range(0,len(dataset),batch_size):
        rows=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
        batch={k:v.to(device) for k,v in collate(rows,model.cfg.pad_token_id).items()}
        logits=model(**{k:batch[k] for k in INPUTS}).float().cpu().tolist()
        for row,z in zip(rows,logits):
            n=len(row["target_probs"]);m=metrics(softmax(z[:n]),row["target_probs"],row["values"])
            count+=1;stats=by_type.setdefault(row["type"],{"n":0,"nll_sum":0.0});stats["n"]+=1;stats["nll_sum"]+=m["nll"]
            lang=language_key(row.get("language"));ls=by_language.setdefault(lang,{"count":0,"nll_sum":0.0,"correct":0,"hard_count":0})
            ls["count"]+=1;ls["nll_sum"]+=m["nll"]
            if "correct" in m:ls["correct"]+=m["correct"];ls["hard_count"]+=1
            for key in ("nll","brier_to_target"):totals[key]=totals.get(key,0.0)+m[key]
            if "correct" in m:hard+=1;correct+=m["correct"]
    if not count:raise ValueError("Validation set is empty")
    return {**{k:v/count for k,v in totals.items()},"count":count,"hard_label_count":hard,
            "accuracy":correct/hard if hard else None,"by_type":by_type,
            "by_language":{k:{**v,"nll":v["nll_sum"]/v["count"],"accuracy":v["correct"]/v["hard_count"] if v["hard_count"] else None} for k,v in by_language.items()},
            "macro_language_nll":sum(v["nll_sum"]/v["count"] for v in by_language.values())/len(by_language),
            "english_nll":by_language.get("en",{}).get("nll_sum",0)/max(1,by_language.get("en",{}).get("count",0)) if "en" in by_language else float("inf")}


def train(args):
    # Defaults preserve 0.2 callers and checkpoint tests; new options are explicit.
    args.language_weights=validate_language_weights(getattr(args,"language_weights",{}))
    args.pair_consistency=float(getattr(args,"pair_consistency",0.0))
    args.selection_metric=getattr(args,"selection_metric","nll")
    args.train_language=getattr(args,"train_language",None)
    args.validation_limit=int(getattr(args,"validation_limit",0))
    args.guard_english=bool(getattr(args,"guard_english",False))
    args.guard_korean=bool(getattr(args,"guard_korean",False))
    if args.guard_korean and not args.guard_english:raise ValueError("Korean guard requires the shared English reference")
    args.english_accuracy_drop=float(getattr(args,"english_accuracy_drop",0.0))
    args.english_nll_increase=float(getattr(args,"english_nll_increase",0.0))
    if args.validation_limit<0 or min(args.english_accuracy_drop,args.english_nll_increase)<0:
        raise ValueError("Invalid validation/guard settings")
    if not math.isfinite(args.pair_consistency) or args.pair_consistency<0:raise ValueError("Invalid pair weight")
    if args.selection_metric not in {"nll","macro_language_nll","english_nll"}:raise ValueError("Unknown selection metric")
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA unavailable")
    if args.bf16 and (args.device!="cuda" or not torch.cuda.is_bf16_supported()):raise ValueError("BF16 requires supported CUDA hardware")
    if min(args.batch_size,args.accumulation,args.epochs,args.eval_every,args.save_every,args.log_every,args.threads)<1:raise ValueError("Invalid positive training argument")
    if args.patience<1 or not 0<args.layer_decay<=1 or min(args.encoder_lr,args.head_lr)<=0:raise ValueError("Invalid optimizer configuration")
    if not 0<=args.warmup_ratio<=1 or min(args.brier_weight,args.rps_weight,args.weight_decay)<0:
        raise ValueError("Invalid warmup/loss/regularization settings")
    model=read_model(args.init)
    source_info=json.loads((Path(args.init)/"init.json").read_text())
    dataset=CacheDataset(args.cache,"train");val=CacheDataset(args.cache,"validation")
    if not len(dataset) or not len(val):raise ValueError("Train and validation splits must be nonempty")
    dataset.verify();val.verify()
    dataset=training_view(dataset,args.train_language)
    val=validation_view(val,args.validation_limit,args.seed)
    fingerprint=dataset.manifest["fingerprint"]
    if source_info.get("data_fingerprint") not in (None,fingerprint):
        same_groups=source_info.get("group_assignment_sha256")==dataset.manifest.get("group_assignment_sha256")
        same_inputs=source_info.get("input_files_sha256")==dataset.manifest.get("input_files_sha256")
        if not (same_groups and same_inputs):
            raise ValueError("Different data exposure: identical raw inputs and group assignments are required for a length-curriculum warm start")
    expected_tokenizer=source_info.get("tokenizer_sha256")
    if expected_tokenizer and dataset.manifest["tokenizer_sha256"]!=expected_tokenizer:raise ValueError("Tokenizer/cache mismatch")
    out=Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError("Use a new output directory or --resume")
    out.mkdir(parents=True,exist_ok=True)
    model.encoder.gradient_checkpointing=args.gradient_checkpointing
    model.to(args.device)
    optimizer=torch.optim.AdamW(parameter_groups(model,args.encoder_lr,args.head_lr,args.layer_decay,args.weight_decay))
    lengths=np.memmap(Path(args.cache)/"train.lengths",dtype="<u4",mode="r")
    if isinstance(dataset,DatasetView):lengths=lengths[dataset.indices]
    units=reviewed_pair_units(dataset) if args.pair_consistency else None
    batches=math.ceil(len(dataset)/args.batch_size);steps_per_epoch=math.ceil(batches/args.accumulation)
    if units is not None:
        total_steps=sum(math.ceil(sum(1 for _ in paired_batches(units,lengths,args.batch_size,args.seed+ep))/args.accumulation) for ep in range(args.epochs))
    else:total_steps=steps_per_epoch*args.epochs
    warmup=max(1,round(total_steps*args.warmup_ratio))
    signature={k:getattr(args,k) for k in ("batch_size","accumulation","epochs","encoder_lr","head_lr","layer_decay",
                "weight_decay","seed","bf16","gradient_checkpointing","warmup_ratio","brier_weight","rps_weight",
                "language_weights","pair_consistency","selection_metric","train_language","validation_limit",
                "guard_english","guard_korean","english_accuracy_drop","english_nll_increase","eval_every","patience")}
    signature["data_fingerprint"]=fingerprint
    epoch=0;offset=0;step=0;best=float("inf");stale=0;guard_reference=None;guard_passed_updates=0
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=True)
        if ck["signature"]!=signature:raise ValueError("Resume settings or data differ; exact optimizer resume refused")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        epoch,offset,step,best,stale=(ck[k] for k in ("epoch","next_batch","step","best","stale"))
        guard_reference=ck.get("guard_reference");guard_passed_updates=ck.get("guard_passed_updates",0)
        random.setstate(ck["python_rng"]);torch.set_rng_state(ck["torch_rng"])
        if args.device=="cuda":torch.cuda.set_rng_state_all(ck["cuda_rng"])
    info={**source_info,"data_fingerprint":fingerprint,"train_rows":len(dataset),"validation_rows":len(val),
          "group_assignment_sha256":dataset.manifest["group_assignment_sha256"],"input_files_sha256":dataset.manifest["input_files_sha256"],
          "max_state_tokens":dataset.manifest["max_state_tokens"],"max_pair_tokens":dataset.manifest["max_pair_tokens"],
          "training_method":"supervised NLL+Brier+ordinal CRPS; optional reviewed-pair JS; no model API calls (teacher-origin input labels remain separate)", "training_signature":signature,"validation_view_rows":len(val)}
    if args.guard_english and guard_reference is None:
        guard_reference=validate(model,val,args.device,args.batch_size)
        english_gate(guard_reference,guard_reference)
        best=guard_reference[args.selection_metric]
        save_inference_checkpoint(model,out/"best",args.init,{**info,"selection_origin":"english_reference_fallback"})
        (out/"english_reference_metrics.json").write_text(json.dumps(guard_reference,indent=2))
    interrupted=[False]
    old_handlers={}
    def request_stop(signum,frame):
        interrupted[0]=True
        print(json.dumps({"stage":"checkpoint_requested","message":"Finishing the current optimizer step before saving."}),flush=True)
    if __import__("threading").current_thread() is __import__("threading").main_thread():
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_handlers[sig]=signal.signal(sig,request_stop)
    def snapshot(ep,next_batch):
        atomic_save({"format":"nd2-training-v1","model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "epoch":ep,"next_batch":next_batch,"step":step,"best":best,"stale":stale,"signature":signature,
            "guard_reference":guard_reference,"guard_passed_updates":guard_passed_updates,
            "python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all() if args.device=="cuda" else []},out/"last.pt")
    start_time=time.perf_counter();trained_rows=0;history=[];stop=False;pair_count=0;language_exposure={}
    optimizer.zero_grad(set_to_none=True)
    for ep in range(epoch,args.epochs):
        # Reconstructible shuffle; only row offsets are in RAM, not tokenized texts.
        order=np.arange(len(dataset),dtype=np.int64);np.random.default_rng(args.seed+ep).shuffle(order)
        for lo in range(0,len(order),2048):
            section=order[lo:lo+2048];order[lo:lo+2048]=section[np.argsort(lengths[section],kind="stable")]
        epoch_batches=list(paired_batches(units,lengths,args.batch_size,args.seed+ep)) if units is not None else None
        batches=len(epoch_batches) if epoch_batches is not None else math.ceil(len(dataset)/args.batch_size)
        accumulated=0;weight_sum=0.0;loss_total=0.0
        model.train()
        first=offset if ep==epoch else 0
        for bi in range(first,batches):
            indices=epoch_batches[bi] if epoch_batches is not None else order[bi*args.batch_size:(bi+1)*args.batch_size]
            rows=[dataset[int(i)] for i in indices]
            for row in rows:
                lang=language_key(row.get("language"));row["weight"]*=args.language_weights.get(lang,1.0)
                e=language_exposure.setdefault(lang,{"rows":0,"effective_weight":0.0});e["rows"]+=1;e["effective_weight"]+=row["weight"]
            batch={k:v.to(args.device) for k,v in collate(rows,model.cfg.pad_token_id,shuffle=True).items()}
            with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=args.bf16):
                logits=model(**{k:batch[k] for k in INPUTS})
                loss,_=proper_loss(logits,batch["target"],batch["option_mask"],batch["values"],batch["ordinal_mask"],args.brier_weight,args.rps_weight)
                weighted=(loss*batch["weight"]).sum()
                if args.pair_consistency:
                    consistency,n_pairs=bilingual_js(logits,rows,batch["original_order"])
                    weighted=weighted+args.pair_consistency*consistency*batch["weight"].sum()
                    pair_count+=n_pairs
            if not torch.isfinite(weighted):raise RuntimeError("Non-finite loss")
            weighted.backward();accumulated+=1;weight_sum+=float(batch["weight"].sum());loss_total+=float(weighted.detach());trained_rows+=len(rows)
            if accumulated<args.accumulation and bi+1<batches:continue
            for p in model.parameters():
                if p.grad is not None:p.grad.div_(weight_sum)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            next_step=step+1
            multiplier=min(1.0,next_step/warmup) if next_step<=warmup else 0.1+0.45*(1+math.cos(math.pi*(next_step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:group["lr"]=group["initial_lr"]*multiplier
            optimizer.step();optimizer.zero_grad(set_to_none=True);step=next_step
            entry={"step":step,"epoch":ep+1,"train_loss":loss_total/weight_sum}
            accumulated=0;weight_sum=0.0;loss_total=0.0
            evaluate=(step%args.eval_every==0 or bi+1==batches)
            if evaluate:
                result=validate(model,val,args.device,args.batch_size);entry["validation"]=result
                eligible=not args.guard_english or english_gate(result,guard_reference,args.english_accuracy_drop,args.english_nll_increase)
                entry["english_gate_passed"]=eligible
                if args.guard_korean:eligible=eligible and korean_gate(result,guard_reference)
                entry["all_requested_language_gates_passed"]=eligible
                if eligible and result[args.selection_metric]<best-1e-8:
                    best=result[args.selection_metric];stale=0;guard_passed_updates+=1
                    save_inference_checkpoint(model,out/"best",args.init,{**info,"selection_origin":"mixed_candidate" if args.guard_english else "validation_best"})
                else:stale+=1
                model.train()
            if step%args.log_every==0 or evaluate:
                print(json.dumps(entry),flush=True);history.append(entry)
                with (out/"history.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(entry)+"\n")
            next_ep,next_batch=(ep+1,0) if bi+1==batches else (ep,bi+1)
            if evaluate or step%args.save_every==0 or (args.stop_after and step>=args.stop_after) or interrupted[0]:snapshot(next_ep,next_batch)
            if stale>=args.patience or (args.stop_after and step>=args.stop_after) or interrupted[0]:stop=True;break
        offset=0
        if stop:break
    save_inference_checkpoint(model,out/"final",args.init,info)
    report={"wall_seconds_this_invocation":time.perf_counter()-start_time,"optimizer_steps_total":step,
        "trained_rows_this_invocation":trained_rows,
        "rows_per_second_this_invocation":trained_rows/max(1e-9,time.perf_counter()-start_time),
        "completed":not interrupted[0] and (step>=total_steps or stale>=args.patience),
        "interrupted":interrupted[0],"english_guard_enabled":args.guard_english,
        "english_gate_accepted_updates":guard_passed_updates,"english_reference_metrics":guard_reference,
        "validation_view_rows":len(val),"best_selection_loss":best if math.isfinite(best) else None,
        "selection_metric":args.selection_metric,"reviewed_pair_comparisons":pair_count,"language_exposure":language_exposure,
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if args.device=="cuda" else None,
        "korean_guard_enabled":args.guard_korean,"device":args.device,"history":history,"quality_status":"Task performance must be measured on the untouched test split"}
    (out/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    for sig,handler in old_handlers.items():signal.signal(sig,handler)
    dataset.close();val.close()
    return report
