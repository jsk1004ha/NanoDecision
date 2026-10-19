import json
from pathlib import Path
from .runtime import DecisionEngine,sha256
from .probability import softmax,metrics,fit_temperature,conformal_quantile,EvaluationAccumulator
from .data import CacheDataset
from .multilingual import language_key,probability_bucket,served_temperature


def evaluate_cache(bundle,cache,split,out):
    engine=DecisionEngine(bundle,require_calibration=False);dataset=CacheDataset(cache,split)
    dataset.verify();fingerprint=dataset.manifest["fingerprint"]
    train_fp=engine.manifest["provenance"].get("data_fingerprint")
    if train_fp and train_fp!=fingerprint:raise ValueError("Different dataset fingerprint: audit cross-dataset leakage before evaluation")
    if engine.manifest["max_state_tokens"]<dataset.manifest["max_state_tokens"] or engine.manifest["max_pair_tokens"]<dataset.manifest["max_pair_tokens"]:
        raise ValueError("Cache token limits exceed exported model limits")
    expected=engine.manifest.get("provenance",{}).get("tokenizer_sha256")
    if expected and expected!=dataset.manifest["tokenizer_sha256"]:raise ValueError("Tokenizer mismatch")
    output=Path(out);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError("Predictions output exists")
    aggregate=EvaluationAccumulator(engine.calibration.get("qhat") if engine.calibration else None);by_type={};by_language={}
    # Calibration inputs must be raw logits from the *deployed quantized model*.
    t=1.0
    with output.open("w",encoding="utf-8") as f:
        for i in range(len(dataset)):
            row=dataset[i]
            context=engine.native.prepare(engine.handle,row["state_ids"])
            z=engine.native.score(engine.handle,context,row["pair_ids"])
            t=served_temperature(engine.calibration,row.get("language"),row["type"],len(z));p=softmax(z,t)
            result={k:row[k] for k in ("id","group_id","type","split","option_ids","target_probs","values")}
            result.update(language=row.get("language","unknown"),task=row.get("task",row["type"]),
                label_origin=row.get("label_origin","unspecified"),reference_option_id=row.get("reference_option_id"),
                temperature_used=t,logits=z,probabilities=p,model_sha256=engine.manifest["model_sha256"],data_fingerprint=fingerprint)
            f.write(json.dumps(result,ensure_ascii=False)+"\n")
            m=metrics(p,row["target_probs"],row["values"])
            aggregate.add(p,row["target_probs"],row["values"],row["group_id"])
            la=by_language.setdefault(language_key(row.get("language")),EvaluationAccumulator(engine.calibration.get("qhat") if engine.calibration else None))
            la.add(p,row["target_probs"],row["values"],row["group_id"])
            stat=by_type.setdefault(row["type"],{"n":0,"correct":0,"hard_n":0,"nll_sum":0.0})
            stat["n"]+=1;stat["nll_sum"]+=m["nll"]
            if "correct" in m:stat["correct"]+=m["correct"];stat["hard_n"]+=1
    summary=aggregate.summary()
    summary.update(by_type=by_type,by_language={k:v.summary() for k,v in by_language.items()},
                   model_sha256=engine.manifest["model_sha256"],split=split,smoke_only=engine.manifest.get("smoke_only",False))
    output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def read_predictions(path):
    with open(path,encoding="utf-8") as f:return [json.loads(line) for line in f if line.strip()]


def calibrate_bundle(bundle,temperature_path,conformal_path,alpha=0.1,language_calibration=False,min_bucket_groups=200):
    bundle=Path(bundle);manifest=json.loads((bundle/"manifest.json").read_text())
    model_hash=sha256(bundle/"model.nd2")
    if manifest["model_sha256"]!=model_hash:raise ValueError("Model modified after export")
    temp=read_predictions(temperature_path);conf=read_predictions(conformal_path)
    if not temp or not conf:raise ValueError("Both calibration sets must be nonempty")
    expected_fp=manifest.get("provenance",{}).get("data_fingerprint")
    for name,rows in (("temperature",temp),("conformal",conf)):
        for r in rows:
            if r["split"]!=name or r["model_sha256"]!=model_hash:raise ValueError("Wrong split/model supplied for calibration")
            if expected_fp and r["data_fingerprint"]!=expected_fp:raise ValueError("Calibration cache is not the training-audited cache")
        if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate calibration IDs")
    if {r["group_id"] for r in temp}&{r["group_id"] for r in conf}:raise ValueError("Temperature/conformal group leakage")
    fitted=fit_temperature(temp)
    if min_bucket_groups<2:raise ValueError("Need at least two independent groups per calibration bucket")
    if language_calibration:
        buckets={}
        for r in temp:
            key=probability_bucket(r.get("language"),r["type"],len(r["logits"]))
            buckets.setdefault(key,[]).append(r)
        table={};counts={}
        for key,rows in buckets.items():
            n_groups=len({r["group_id"] for r in rows});counts[key]=n_groups
            if n_groups>=min_bucket_groups:
                value=fit_temperature(rows)
                # The global temperature is also an explicit fit-set fallback.
                global_loss=sum(metrics(softmax(x["logits"],fitted["temperature"]),x["target_probs"])["nll"] for x in rows)/len(rows)
                if value["after_nll"]>global_loss:value.update(temperature=fitted["temperature"],after_nll=global_loss)
                table[key]={**value,"independent_groups":n_groups}
        fitted.update(temperatures_by_bucket=table,bucket_group_counts=counts,
                      min_bucket_groups=min_bucket_groups,language_source="Explicit row/API metadata, not confidence-based language guessing")
    # One nonconformity score per source group. Using the maximum avoids
    # treating correlated questions from one state as independent calibration data.
    groups={}
    for r in conf:
        if max(r["target_probs"])<1-1e-7:
            raise ValueError("Conformal calibration requires hard labels; soft labels cannot establish label coverage")
        p=softmax(r["logits"],served_temperature(fitted,r.get("language"),r["type"],len(r["logits"])))
        truth=r["target_probs"].index(max(r["target_probs"]))
        score=1-p[truth]
        groups[r["group_id"]]=max(groups.get(r["group_id"],0.0),score)
    scores=list(groups.values())
    result={**fitted,"qhat":conformal_quantile(scores,alpha),"alpha":alpha,"model_sha256":model_hash,
        "temperature_input_sha256":sha256(temperature_path),"conformal_input_sha256":sha256(conformal_path),
        "conformal_groups":len(scores),"conformal_rows_supplied":len(conf),"method":"split group-max LAC",
        "guarantee_scope":"Marginal simultaneous label-set coverage for a fresh group under exchangeability of complete groups with the same question-sampling process; not per-domain, pointwise, or accepted-only accuracy.",
        "caveat":"Calibration does not detect arbitrary distribution shift or prove a chosen action is safe."}
    (bundle/"calibration.json").write_text(json.dumps(result,indent=2))
    return result
