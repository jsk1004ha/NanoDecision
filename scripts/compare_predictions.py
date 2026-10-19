#!/usr/bin/env python3
"""Two- or multi-model comparison; no silently dropped requests or stale input."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from nanodecision2.benchmark import align_records,summarize,cluster_bootstrap_delta


def load(path):
    rows={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if r["id"] in rows:raise ValueError("Duplicate ID")
            rows[r["id"]]=r
    if not rows:raise ValueError("No predictions")
    return rows


def compare(paths,repeats=2000,seed=42):
    models={Path(p).stem:load(p) for p in paths}
    if len(models)!=len(paths):raise ValueError("Use distinct prediction filenames")
    keys=list(models);first=models[keys[0]]
    for other in list(models.values())[1:]:
        if set(first)!=set(other):raise ValueError("Different request sets: missing/error rows must be recorded, not dropped")
        for rid in first:align_records(first[rid],other[rid])
    common=[rid for rid in first if all(x[rid].get("status","ok")=="ok" for x in models.values())]
    full={name:summarize(list(rows.values())) for name,rows in models.items()}
    report={"all_requested":len(first),"common_supported":len(common),"common_fraction":len(common)/len(first),
            "all_requests":full,"common_supported_metrics":{},"pairwise":{},"by_language":{},"by_task":{}}
    if not common:
        report["note"]="No shared supported inputs; no model quality comparison possible. See failure counts.";return report
    for name,rows in models.items():report["common_supported_metrics"][name]=summarize([rows[i] for i in common])
    for index,(a,b) in enumerate(combinations(keys,2)):
        report["pairwise"][f"{b}-minus-{a}"]=cluster_bootstrap_delta([models[a][i] for i in common],[models[b][i] for i in common],repeats,seed+index)
    for field in ("language","task"):
        slices=defaultdict(list)
        for rid in common:slices[first[rid].get(field,"unknown")].append(rid)
        report["by_"+field]={value:{name:summarize([rows[i] for i in ids]) for name,rows in models.items()} for value,ids in slices.items()}
    report["pairwise_by_language"]={}
    langs=sorted({first[i].get("language","unknown") for i in common})
    for lang in langs:
        ids=[i for i in common if first[i].get("language","unknown")==lang]
        report["pairwise_by_language"][lang]={}
        for index,(a,b) in enumerate(combinations(keys,2)):
            report["pairwise_by_language"][lang][f"{b}-minus-{a}"]=cluster_bootstrap_delta([models[a][i] for i in ids],[models[b][i] for i in ids],repeats,seed+index)
    report["interpretation"]="Quality deltas use the SAME common supported rows across all models. Coverage and failure reasons remain visible. Do not call teacher-label agreement factual correctness; do not compare different device latency or declare universal superiority."
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",nargs="+");p.add_argument("--a");p.add_argument("--b")
    p.add_argument("--out",required=True);p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();paths=a.predictions or ([a.a,a.b] if a.a and a.b else [])
    if len(paths)<2:p.error("Supply at least two --predictions files or --a and --b")
    out=Path(a.out)
    if out.exists():raise FileExistsError("Report exists")
    out.parent.mkdir(parents=True,exist_ok=True)
    report=compare(paths,a.bootstrap,a.seed);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
