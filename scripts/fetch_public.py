#!/usr/bin/env python3
"""Convert original public TRAIN splits, no Jev calls and no row inflation.

Installs: pip install '.[data]'. Public downloads require network access; xLAM
currently requires HF terms acceptance. Always inspect the output manifest.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata


def h(s):return hashlib.sha256(unicodedata.normalize("NFKC",s).strip().encode()).hexdigest()


def main():
    from datasets import load_dataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",choices=("snli","multinli","klue-nli","xlam"),required=True)
    p.add_argument("--out",required=True);p.add_argument("--revision",default="main");p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    configs={"snli":("stanfordnlp/snli",None),"multinli":("nyu-mll/multi_nli",None),
             "klue-nli":("klue/klue","nli"),"xlam":("Salesforce/xlam-function-calling-60k",None)}
    name,config=configs[args.source]
    dataset=load_dataset(name,config,split="train",revision=args.revision,streaming=True)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError("Output already exists")
    stats=Counter();ko=args.source=="klue-nli"
    descriptions={"entailment":"전제가 참이라면 가설이 반드시 성립한다.","neutral":"전제만으로 가설의 참과 거짓을 결정할 수 없다.","contradiction":"전제가 참이라면 가설은 거짓이다."} if ko else {
        "entailment":"The hypothesis must be true if the premise is true.",
        "neutral":"The premise does not establish whether the hypothesis is true or false.",
        "contradiction":"The hypothesis must be false if the premise is true."}
    labels=None if args.source=="xlam" else dataset.features["label"].names
    with path.open("w",encoding="utf-8") as f:
        for i,row in enumerate(dataset):
            if args.limit and stats["written"]>=args.limit:break
            stats["read"]+=1
            if args.source!="xlam":
                label=int(row["label"])
                if label<0:stats["missing_label"]+=1;continue
                target=labels[label]
                if target not in descriptions:raise ValueError(f"Unknown source label {target}")
                premise,hypothesis=row["premise"],row["hypothesis"]
                # Group by original premise across sources, not each hypothesis.
                result={"id":args.source+":"+str(i),"group_id":"nli-premise:"+h(premise),
                    "task":"nli","language":"ko" if ko else "en","state":premise,"type":"choice",
                    "question":("다음 가설은 전제와 어떤 관계인가?\n" if ko else "How does this hypothesis relate to the premise?\n")+hypothesis,
                    "criteria":descriptions,"target":target}
            else:
                tools=json.loads(row["tools"]) if isinstance(row["tools"],str) else row["tools"]
                answers=json.loads(row["answers"]) if isinstance(row["answers"],str) else row["answers"]
                # Multi-tool targets must NOT be silently reduced to a wrong single-label target.
                if len(answers)!=1 or not 2<=len(tools)<=256:stats["not_single_choice"]+=1;continue
                query=row["query"]
                if not isinstance(query,str):query=json.dumps(query,ensure_ascii=False)
                names=[t["name"] for t in tools];target=answers[0]["name"]
                if len(set(names))!=len(names) or target not in names:stats["invalid_tools"]+=1;continue
                result={"id":"xlam:"+str(row.get("id",i)),"group_id":"tool-query:"+h(query),"task":"tool_route",
                    "language":"en","state":query,"type":"choice","question":"Which single tool should handle this request?",
                    "choices":[{"id":t["name"],"description":t["description"]+"\nParameters: "+json.dumps(t["parameters"],ensure_ascii=False,sort_keys=True)} for t in tools],
                    "target":target,"synthetic_source":True}
            result["source"]={"dataset":name,"revision":args.revision,"original_split":"train","original_row":i}
            f.write(json.dumps(result,ensure_ascii=False)+"\n");stats["written"]+=1
    from nanodecision2.runtime import sha256
    report={"source":name,"requested_revision":args.revision,"original_split":"train","counts":dict(stats),
            "sha256":sha256(path),"note":"No augmentation. Output rows remain distinct from raw-source count. Use a pinned revision for reproducibility."}
    path.with_suffix(".manifest.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
