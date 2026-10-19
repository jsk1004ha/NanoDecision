#!/usr/bin/env python3
"""Run ND2, Julia 1, or an explicit Laya checkpoint in an isolated environment.

One JSONL row per request, including unsupported/truncated/error rows. Laya's
strict preflight compares the complete rendered sequence against what its own
builder actually encodes. No gold-assisted candidate shortlist or auto-routing.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from nanodecision2.schema import normalize_question
from nanodecision2.benchmark import make_record,summarize
from nanodecision2.probability import check_distribution,softmax
from nanodecision2.runtime import sha256


class UnsupportedInput(ValueError):pass


def laya_predictor(agent,max_length,head_length,raw_logits=False):
    import torch
    from laya.common import build_sequence,collate_items,QTYPES,render_options,temp_bucket
    def predict(row):
        spec=normalize_question(row)
        criteria=[o["description"] for o in spec["options"]] if spec["type"]=="score" else {o["id"]:o["description"] for o in spec["options"]}
        q={"t":spec["type"],"ins":spec["question"],"crit":criteria}
        tok=agent.tok
        def enc(s):return tok(s,add_special_tokens=False)["input_ids"]
        from laya.common import serialize_state
        st=serialize_state(row["state"])
        texts=render_options(q)
        if any(tok.mask_token in x for x in [st,spec["question"]]+texts):
            raise UnsupportedInput("Laya sanitizes literal mask tokens; common unmodified-input track rejects this input")
        fullhead=enc(f"{q['t']} question: {q['ins']}")
        opts=[enc(" "+x) for x in texts]
        if any(len(x)>48 for x in opts):raise UnsupportedInput("Laya per-option token cap would truncate an option")
        expected=[tok.cls_token_id]+fullhead+[tok.sep_token_id]
        for o in opts:expected += [tok.mask_token_id]+o
        expected += [tok.sep_token_id]+enc(st)+[tok.sep_token_id]
        ids,markers,stats=build_sequence(tok,row["state"],q,max_length,head_length,return_stats=True)
        if ids!=expected or len(markers)!=len(opts) or stats["options_distinct"]!=len(opts):
            raise UnsupportedInput("Laya head/state/options were truncated or collapsed; refused silently changed input")
        b=collate_items([[{"ids":ids,"markers":markers,"qtype":QTYPES[q["t"]]}]],tok.pad_token_id)
        with torch.inference_mode():
            logits,_=agent.model(b["input_ids"].to(agent.device),b["attention_mask"].to(agent.device),
                b["marker_pos"].to(agent.device),b["marker_mask"].to(agent.device),b["qtype"].to(agent.device))
        z=logits[0,:len(opts)].float().cpu().tolist();qt=QTYPES[q["t"]]
        # Explicit as-shipped probability calibration; learned act/entropy is
        # not treated as a calibrated probability of correctness.
        t=1.0 if raw_logits else float(agent.temperature_by_options.get(temp_bucket(qt,len(opts)),agent.temperature[qt]))
        return {"probabilities":softmax(z,t),"logits":z,"temperature_used":t,"encoded_tokens":len(ids),"strict_input_verified":True}
    return predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",choices=("nd2","julia","laya"),required=True)
    p.add_argument("--model",required=True);p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--threads",type=int,default=4);p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--head-length",type=int,default=512)
    p.add_argument("--julia-max-length",type=int,help="Deprecated alias")
    p.add_argument("--laya-variant",choices=("multilingual","english","typed-decisions"),default="multilingual")
    p.add_argument("--probability-mode",choices=("shipped","unscaled"),default="shipped")
    p.add_argument("--warmup",type=int,default=2)
    args=p.parse_args()
    import os
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key]=str(args.threads)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError("Predictions output exists")
    if min(args.threads,args.max_length,args.head_length)<1 or args.warmup<0:raise ValueError("Invalid run limits")
    if args.julia_max_length:args.max_length=args.julia_max_length
    if args.engine=="julia" and args.probability_mode!="shipped":raise ValueError("Julia named API exposes shipped probabilities; no undocumented de-calibration")
    os.environ.setdefault("USE_TF","0");os.environ["JULIA_CPU_THREADS"]=str(args.threads)
    engine_name=args.engine if args.engine!="laya" else "laya-"+args.laya_variant
    start=time.perf_counter();modelpath=Path(args.model)
    if not modelpath.is_dir():raise ValueError("Use an explicit downloaded local model directory, not an auto-routed hub ID")
    model_files={str(f.relative_to(modelpath)):sha256(f) for f in sorted(modelpath.rglob("*"))
                 if f.is_file() and f.suffix in {".safetensors",".nd2",".json"} and ".cache" not in f.parts}
    if args.engine=="nd2":
        if args.device!="cpu":raise ValueError("ND2 native backend is CPU-only")
        from nanodecision2 import DecisionEngine
        engine=DecisionEngine(args.model,require_calibration=False)
        def predict(row):
            spec=normalize_question(row);answer=engine.logits_row(row)
            p=[answer["probabilities"][o["id"]] for o in spec["options"]]
            if args.probability_mode=="unscaled":p=softmax(answer["logits"]);answer["temperature_used"]=1.0
            return {"probabilities":p,"logits":answer["logits"],"temperature_used":answer["temperature_used"],"strict_input_verified":True}
    else:
        import torch
        torch.set_num_threads(args.threads)
        if args.engine=="julia":
            from julia import load_model
            engine=load_model(args.model,device=args.device,strict_encoding=True,max_length=args.max_length,head_length=args.head_length)
            def predict(row):
                spec=normalize_question(row);opts=spec["options"]
                if len(opts)>20:raise UnsupportedInput("Julia native supports at most 20 options; no oracle shortlist allowed")
                criteria=[x["description"] for x in opts] if spec["type"]=="score" else {x["id"]:x["description"] for x in opts}
                request={"type":spec["type"],"instructions":spec["question"],"criteria":criteria}
                result=engine.predict(state=row["state"],questions={"q":request})["answers"]["q"]
                return {"probabilities":[result["probabilities"][x["id"]] for x in opts],"strict_input_verified":True}
        else:
            import laya
            engine=laya.load(args.model,device=args.device);engine.model.eval()
            predict=laya_predictor(engine,args.max_length,args.head_length,args.probability_mode=="unscaled")
    load_seconds=time.perf_counter()-start;records=[];seen=set();warm=False
    def sync():
        if args.device=="cuda":
            import torch
            torch.cuda.synchronize()
    with open(args.input,encoding="utf-8") as src,out.open("w",encoding="utf-8") as dst:
        for line in src:
            if not line.strip():continue
            row=json.loads(line);record=make_record(row,engine_name)
            if row["id"] in seen:raise ValueError("Duplicate evaluation ID")
            seen.add(row["id"])
            try:
                if not warm:
                    for _ in range(args.warmup):predict(row)
                    sync();warm=True
                sync();t=time.perf_counter();answer=predict(row);sync()
                elapsed=time.perf_counter()-t;check_distribution(answer["probabilities"])
                if len(answer["probabilities"])!=len(record["option_ids"]):raise ValueError("Prediction length mismatch")
                record.update(answer,status="ok",seconds=elapsed)
            except (ValueError,RuntimeError,KeyError) as e:
                record.update(status="failed",error_kind="unsupported_input" if isinstance(e,UnsupportedInput) else type(e).__name__,error=str(e))
            records.append(record);dst.write(json.dumps(record,ensure_ascii=False)+"\n");dst.flush()
    report=summarize(records);times=sorted(r["seconds"] for r in records if r["status"]=="ok")
    def quantile(frac):
        if not times:return None
        x=(len(times)-1)*frac;lo=int(x);hi=min(lo+1,len(times)-1)
        return times[lo]+(times[hi]-times[lo])*(x-lo)
    from importlib import metadata
    versions={}
    for package in ("nanodecision2","torch","transformers","tokenizers","laya","julia"):
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:pass
    report.update(package_versions=versions,engine=engine_name,device=args.device,probability_mode=args.probability_mode,input_sha256=sha256(args.input),
        model_files_sha256=model_files,limits={"max_length":args.max_length,"head_length":args.head_length},
        timing={"load_and_fingerprint_seconds":load_seconds,"warmup_requests":args.warmup,"p50_seconds":quantile(.5),"p95_seconds":quantile(.95),
                "scope":"One question per request; includes encoding and decoding, excludes model load, file IO and warmup. CPU/native and GPU are separate tracks."},
        python=sys.version,platform=platform.platform(),requested_threads=args.threads,
        smoke_only=bool(engine.manifest.get("smoke_only",False)) if args.engine=="nd2" else False)
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"]=rss if sys.platform=="darwin" else rss*1024
        report["rss_scope"]="Whole process high-water mark including model loading, hashing and prediction bookkeeping; not just tensor storage"
    except ImportError:report["process_peak_rss_bytes"]=None
    out.with_suffix(".summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
