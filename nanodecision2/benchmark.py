"""Matched-input evaluation helpers, including explicit failure accounting."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from .schema import normalize_question, gold_target, state_text
from .probability import check_distribution, metrics
from .multilingual import language_key


def input_fingerprint(row):
    spec=normalize_question(row)
    payload={"state":row["state"],"type":spec["type"],"question":spec["question"],
             "options":spec["options"],"values":spec["values"]}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def make_record(row,engine):
    spec=normalize_question(row);target=gold_target(row,spec)
    names=[o["id"] for o in spec["options"]]
    label=row.get("reference_option_id")
    if label is None and max(target)>1-1e-7:label=names[target.index(max(target))]
    if label is not None and label not in names:raise ValueError("Invalid explicit reference label")
    return {"id":row["id"],"group_id":hashlib.sha256(row["group_id"].encode()).hexdigest(),
        "input_sha256":input_fingerprint(row),"engine":engine,"type":spec["type"],"option_ids":names,
        "target_probs":target,"values":spec["values"],"reference_option_id":label,
        "language":language_key(row.get("language")),"task":row.get("task",spec["type"]),
        "label_origin":row.get("label_origin","unspecified"),"source":row.get("source","unknown"),
        "prompt_language":row.get("prompt_language","unspecified"),"status":"pending"}


def record_metrics(row):
    if row.get("status","ok")!="ok":raise ValueError("Failed predictions do not have probabilities")
    p=row["probabilities"];q=row["target_probs"];out=metrics(p,q,row.get("values"))
    pred=max(range(len(p)),key=p.__getitem__)
    out["agreement_to_target_mode"]=float(pred==max(range(len(q)),key=q.__getitem__))
    out["soft_accuracy"]=q[pred]
    ref=row.get("reference_option_id")
    if ref is not None:out["accuracy"]=float(row["option_ids"][pred]==ref)
    elif "correct" in out:out["accuracy"]=float(out["correct"])
    if row.get("values") is not None:
        vals=row["values"];out["score_mean_error"]=abs(sum(x*v for x,v in zip(p,vals))-sum(x*v for x,v in zip(q,vals)))
    return out


def summarize(rows):
    n=len(rows);ok=[r for r in rows if r.get("status","ok")=="ok"]
    if not n:raise ValueError("No evaluation rows")
    values=defaultdict(list);bins=[[0,0.0,0.0] for _ in range(15)]
    for r in ok:
        m=record_metrics(r)
        for k,v in m.items():values[k].append(v)
        if "accuracy" in m:
            c=max(r["probabilities"]);i=min(14,int(c*15));bins[i][0]+=1;bins[i][1]+=c;bins[i][2]+=m["accuracy"]
    result={"requested":n,"predicted":len(ok),"coverage":len(ok)/n,
        "failures":dict(Counter(r.get("error_kind","error") for r in rows if r.get("status","ok")!="ok")),
        "label_origins":dict(Counter(r.get("label_origin","unspecified") for r in rows)),
        "metrics_on_predicted":{k:sum(v)/len(v) for k,v in values.items()}}
    hard_n=sum(x[0] for x in bins)
    result["ece_against_reference_label"]=sum(abs(x[1]-x[2]) for x in bins)/hard_n if hard_n else None
    result["resolved_correct_fraction_of_all"]=sum(values.get("accuracy",[]))/n if hard_n else None
    result["note"]="Reference-label agreement is not necessarily real-world correctness; teacher gold is labeled separately. Failures are never silently dropped."
    return result


def align_records(a,b):
    for key in ("id","group_id","input_sha256","type","values","language","task","label_origin","reference_option_id"):
        if a.get(key)!=b.get(key):raise ValueError(f"Comparison mismatch: {key} at {a.get('id')}")
    if a.get("input_sha256") is None:raise ValueError("Input fingerprints are required; regenerate old predictions")
    if a["option_ids"]!=b["option_ids"] or a["target_probs"]!=b["target_probs"]:
        raise ValueError("Option order/gold mismatch")


def cluster_bootstrap_delta(a,b,repeats=2000,seed=42,alpha=.05):
    """b-a; complete source groups resampled, row-weighted statistic retained."""
    import numpy as np
    if repeats<100 or not 0<alpha<1:raise ValueError("Invalid bootstrap configuration")
    for x,y in zip(a,b):align_records(x,y)
    if len(a)!=len(b) or not a:raise ValueError("Need matched nonempty predictions")
    ma=[record_metrics(x) for x in a];mb=[record_metrics(x) for x in b]
    keys=sorted(set.intersection(*(set(x) for x in ma+mb)) & {"accuracy","nll","brier_to_target","soft_accuracy","normalized_crps","score_mean_error","agreement_to_target_mode"})
    grouped={}
    for row,x,y in zip(a,ma,mb):
        cell=grouped.setdefault(row["group_id"],[0,[0.0]*len(keys)])
        cell[0]+=1
        for i,key in enumerate(keys):cell[1][i]+=y[key]-x[key]
    counts=np.array([v[0] for v in grouped.values()],dtype=float)
    sums=np.array([v[1] for v in grouped.values()],dtype=float)
    point=sums.sum(0)/counts.sum();g=len(counts)
    if g<2:return {"groups":g,"rows":len(a),"direction":"b-minus-a","deltas":dict(zip(keys,point.tolist())),"intervals":None}
    rng=np.random.default_rng(seed);samples=[]
    # Cap allocation for large source-group collections.
    for _ in range(repeats):
        ids=rng.integers(0,g,size=g);samples.append(sums[ids].sum(0)/counts[ids].sum())
    sample=np.asarray(samples);bounds=np.quantile(sample,[alpha/2,1-alpha/2],axis=0)
    return {"groups":g,"rows":len(a),"direction":"b-minus-a","repeats":repeats,"interval_level":1-alpha,
        "deltas":{k:{"value":float(point[i]),"interval":[float(bounds[0,i]),float(bounds[1,i])]} for i,k in enumerate(keys)},
        "note":"Approximate cluster bootstrap. Predeclare primary tests; intervals are not automatically corrected across metrics/tasks."}
