#!/usr/bin/env python3
"""Build versioned typed decisions from original public records; no model API.

MASSIVE uses the official archive directly, avoiding removed HF dataset-script
execution. All sources also accept a local raw JSONL/TSV release via --raw.
"""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import tarfile
import urllib.request
from nanodecision2 import adapters as A
from nanodecision2.runtime import sha256

REGISTRY = {
    "snli":("stanfordnlp/snli",None), "multinli":("nyu-mll/multi_nli",None),
    "klue-nli":("klue/klue","nli"), "klue-ynat":("klue/klue","ynat"),
    "kornli-snli":("kakaobrain/kor_nli","snli"), "kornli-multinli":("kakaobrain/kor_nli","multi_nli"),
    "kornli-xnli":("kakaobrain/kor_nli","xnli"),
    "paws-ko":("google-research-datasets/paws-x","ko"), "paws-en":("google-research-datasets/paws-x","en"),
    "typed-decisions":("LocalLLaMA/typed-decisions","all"),
    "ag-news":("fancyzhx/ag_news",None), "enron-spam":("SetFit/enron_spam",None),
    "phishing":("zefang-liu/phishing-email-dataset",None),
    "support-queue":("Tobi-Bueck/customer-support-tickets",None),
    "msmarco":("microsoft/ms_marco","v1.1")}
MASSIVE_URL="https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"


def read_raw(path):
    with open(path,encoding="utf-8") as f:
        if Path(path).suffix==".tsv":
            yield from csv.DictReader(f,delimiter="\t")
        else:
            for line in f:
                if line.strip():yield json.loads(line)


def massive_archive(path, locale, split):
    # Read a regular member only; never extract arbitrary paths/symlinks.
    with tarfile.open(path,"r:gz") as tar:
        matches=[m for m in tar.getmembers() if m.isfile() and (m.name.lstrip("./")=="data/"+locale+".jsonl" or m.name.endswith("/data/"+locale+".jsonl"))]
        if len(matches)!=1:raise ValueError("Expected exactly one locale file in official MASSIVE archive")
        with io.TextIOWrapper(tar.extractfile(matches[0]),encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                if r["partition"]==split:yield r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",choices=list(REGISTRY)+["massive-scenario","massive-intent"],required=True)
    p.add_argument("--split",default="train");p.add_argument("--language",default="ko-KR")
    p.add_argument("--prompt-language",choices=("ko","en"))
    p.add_argument("--raw");p.add_argument("--revision",default="main")
    p.add_argument("--archive",default="data/raw/massive-1.1.tar.gz")
    p.add_argument("--labels",help="JSON list of 60 intent labels for MASSIVE intent, no oracle shortlisting")
    p.add_argument("--out",required=True);p.add_argument("--limit",type=int,default=0)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError("Output already exists")
    if args.limit<0:raise ValueError("limit must be nonnegative")
    out.parent.mkdir(parents=True,exist_ok=True)
    resolved=None;features={};source_id=args.source
    if args.raw:
        rows=read_raw(args.raw);resolved={"raw_sha256":sha256(args.raw)}
    elif args.source.startswith("massive-"):
        archive=Path(args.archive);archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            temp=Path(str(archive)+".partial")
            with urllib.request.urlopen(MASSIVE_URL,timeout=60) as response,temp.open("wb") as f:
                import shutil
                shutil.copyfileobj(response,f,length=1<<20)
            temp.replace(archive)
        rows=massive_archive(archive,args.language,args.split)
        resolved={"url":MASSIVE_URL,"archive_sha256":sha256(archive)}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        source_id,config=REGISTRY[args.source]
        resolved=HfApi().dataset_info(source_id,revision=args.revision).sha
        rows=load_dataset(source_id,config,split=args.split,revision=resolved,streaming=True)
        features=rows.features or {}
    names=getattr(features.get("label"),"names",None)
    # Known public KLUE and KorNLI class orders; external local NLI should use string labels.
    if args.source.startswith(("klue-nli","kornli-")) and names is None:names=["entailment","neutral","contradiction"]
    labels=json.loads(Path(args.labels).read_text()) if args.labels else None
    count=skipped=seen=0;languages=Counter();parts=Counter();groups=set()
    temp=Path(str(out)+".partial")
    try:
        with temp.open("w",encoding="utf-8") as f:
            for i,r in enumerate(rows):
                seen+=1
                if args.raw and args.source.startswith("massive-") and r["partition"]!=args.split:continue
                if args.source.startswith("massive-"):
                    converted=A.massive_row(r,args.source,labels,args.prompt_language)
                elif args.source in {"snli","multinli","klue-nli"} or args.source.startswith("kornli-"):
                    r=dict(r)
                    if "sentence1" in r:r["premise"]=r["sentence1"];r["hypothesis"]=r["sentence2"]
                    if "gold_label" in r:r["label"]=r["gold_label"]
                    lang="ko" if args.source.startswith(("klue","kornli")) else "en"
                    converted=A.nli_row(r,args.source,args.split,lang,names,original_id=r.get("guid",str(i)))
                    for x in converted:
                        if args.source.startswith("kornli") and args.split=="train":x["label_origin"]="machine_translated_train"
                elif args.source.startswith("paws-"):
                    converted=A.paws_row(r,args.split,args.source.split("-")[-1])
                elif args.source=="typed-decisions":converted=A.typed_rows(r,args.split)
                elif args.source=="msmarco":
                    # is_selected=0 is an unselected passage, NOT a proven irrelevant passage.
                    # Only compare a selected passage against another selected answer? No:
                    # this adapter intentionally emits weak relevance labels and labels provenance.
                    converted=[];group="msmarco:"+str(r.get("query_id",i))
                    for j,(text,label) in enumerate(zip(r["passages"]["passage_text"],r["passages"]["is_selected"])):
                        x=A.base_row("msmarco",args.split,f"{r.get('query_id',i)}:{j}",group,"en")
                        x.update(type="noul",state={"query":r["query"],"passage":text},
                            question="Was this passage selected as supporting evidence for the query?",
                            target=bool(label),task="passage_selection",label_origin="passage_selection_not_exhaustive_relevance")
                        converted.append(x)
                else:converted=A.simple_source_rows(r,args.source,args.split,str(i))
                if not converted:skipped+=1
                for x in converted:
                    x["source_revision"]=resolved
                    f.write(json.dumps(x,ensure_ascii=False)+"\n");count+=1
                    languages[x["language"]]+=1;parts[x["partition"]]+=1;groups.add(x["group_id"])
                if args.limit and seen>=args.limit:break
        temp.replace(out)
    except Exception:
        # Keep partial data clearly named; it must not masquerade as a complete release.
        raise
    manifest={"source":source_id,"requested_revision":args.revision,"resolved":resolved,
              "source_split":args.split,"source_records_seen":seen,"skipped_records":skipped,
              "decision_rows":count,"independent_origin_groups":len(groups),"by_language":languages,
              "by_partition":parts,"output_sha256":sha256(out),
              "limitations":"Schema fixtures tested offline; live downloads and large-scale quality not verified. Review source terms. No automatic million-row quality claim."}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
