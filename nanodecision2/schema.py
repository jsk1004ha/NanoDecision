import json
import math
from .probability import check_distribution


def state_text(state):
    if isinstance(state,str):return state
    return json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def normalize_question(q):
    kind=q.get("type","choice")
    if kind=="bool":kind="noul"
    if kind not in {"choice","noul","score"}:raise ValueError("Unknown question type")
    text=q.get("question",q.get("instructions",""))
    if not isinstance(text,str) or not text.strip():raise ValueError("A nonempty question is required")
    values=None
    if kind=="choice":
        if "criteria" in q:
            if not isinstance(q["criteria"],dict):raise ValueError("choice criteria must be an ID-description mapping")
            options=[{"id":str(k),"description":v} for k,v in q["criteria"].items()]
        else:options=q.get("choices",[])
    elif kind=="noul":
        criteria=q.get("criteria",{})
        options=[{"id":"false","description":criteria.get("false",q.get("false_description","The statement is false."))},
                 {"id":"true","description":criteria.get("true",q.get("true_description","The statement is true."))}]
    else:
        levels=q.get("levels")
        if levels is None:levels=[{"value":i,"description":d} for i,d in enumerate(q.get("criteria",[]))]
        values=[float(x["value"]) for x in levels]
        if not all(math.isfinite(x) for x in values) or any(b<=a for a,b in zip(values,values[1:])):
            raise ValueError("Score levels must be finite and strictly increasing")
        options=[{"id":str(i),"description":x["description"]} for i,x in enumerate(levels)]
    if not 2<=len(options)<=256:raise ValueError("Need 2..256 options")
    if not all(isinstance(o,dict) and isinstance(o.get("id"),str) and isinstance(o.get("description"),str)
               and o["id"] and o["description"].strip() for o in options):
        raise ValueError("Options require nonempty string IDs and descriptions")
    ids=[o["id"] for o in options]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate option IDs")
    if len({o["description"].strip() for o in options})!=len(options):raise ValueError("Duplicate option descriptions")
    return {"type":kind,"question":text,"options":options,"values":values}


def gold_target(row, spec):
    n=len(spec["options"])
    if "target_probs" in row:
        p=[float(x) for x in row["target_probs"]]
        if len(p)!=n:raise ValueError("target_probs length mismatch")
        check_distribution(p);return p
    if "target" not in row:raise ValueError("Training row has no target or target_probs")
    target=row["target"]
    if spec["type"]=="noul":
        if type(target) is not bool:raise ValueError("noul target must be a Boolean; probabilities use target_probs")
        idx=int(target)
    elif spec["type"]=="score":
        if type(target) not in (float,int) or float(target) not in spec["values"]:
            raise ValueError("Score target must be an exact level value; use target_probs for uncertain labels")
        idx=spec["values"].index(float(target))
    else:
        names=[x["id"] for x in spec["options"]]
        if target not in names:raise ValueError("Target is not an option ID")
        idx=names.index(target)
    return [float(i==idx) for i in range(n)]


class TextCodec:
    """Official tokenizer.json through Hugging Face's Rust tokenizers engine.

    Byte codec is available only for the explicitly marked offline smoke model.
    """
    def __init__(self,path=None,smoke=False):
        self.smoke=smoke
        if smoke:self.tokenizer=None
        else:
            from tokenizers import Tokenizer
            self.tokenizer=Tokenizer.from_file(str(path))
            self.tokenizer.no_truncation();self.tokenizer.no_padding()

    def encode(self,text,limit):
        ids=([2]+[b+4 for b in text.encode("utf-8")]+[1]) if self.smoke else self.tokenizer.encode(text,add_special_tokens=True).ids
        if not ids or len(ids)>limit:raise ValueError(f"Input has {len(ids)} tokens; limit is {limit}. No silent truncation.")
        return ids

    def state(self,state,limit):
        return self.encode("STATE\n"+state_text(state),limit)

    def pairs(self,spec,limit):
        # IDs are returned to the caller but never fed to the neural network.
        return [self.encode(f"TYPE: {spec['type']}\nQUESTION:\n{spec['question']}\nCANDIDATE:\n{o['description']}",limit)
                for o in spec["options"]]
