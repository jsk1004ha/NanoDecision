"""Text API: Rust tokenizer + C++ neural network; no torch/numpy imports."""
import hashlib
import json
from pathlib import Path
from .schema import TextCodec, normalize_question
from .probability import softmax,prediction_set,bayes_action
from .multilingual import served_temperature,language_key


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


class DecisionEngine:
    def __init__(self,bundle,require_calibration=True):
        from . import _native
        self.native=_native;self.root=Path(bundle)
        self.manifest=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        self.cfg=self.manifest["config"];self.require_calibration=require_calibration
        self.handle=_native.load(str(self.root/"model.nd2"))
        self.codec=TextCodec(self.root/"tokenizer.json",smoke=self.manifest.get("smoke_only",False))
        path=self.root/"calibration.json";self.calibration=None
        if path.exists():
            self.calibration=json.loads(path.read_text(encoding="utf-8"))
            if self.calibration["model_sha256"]!=sha256(self.root/"model.nd2"):
                raise ValueError("Calibration belongs to another/modified model; recalibrate after quantization")

    def prepare(self,state):
        ids=self.codec.state(state,self.manifest["max_state_tokens"])
        return self.native.prepare(self.handle,ids)

    def score_prepared(self,prepared,question,min_confidence=0.0,language=None):
        if not 0<=min_confidence<=1:raise ValueError("min_confidence must lie in [0,1]")
        spec=normalize_question(question)
        pairs=self.codec.pairs(spec,self.manifest["max_pair_tokens"])
        logits=self.native.score(self.handle,prepared,pairs)
        language=language_key(language or question.get("language"))
        temp=served_temperature(self.calibration,language,spec["type"],len(spec["options"]))
        p=softmax(logits,temp);best=max(range(len(p)),key=p.__getitem__)
        reasons=[];indices=None
        if self.manifest.get("smoke_only"):reasons.append("smoke_model_not_for_use")
        if self.require_calibration and self.calibration is None:reasons.append("uncalibrated")
        if p[best]<min_confidence:reasons.append("low_confidence")
        if self.calibration:
            indices=prediction_set(p,self.calibration["qhat"])
            if len(indices)!=1:reasons.append("non_singleton_prediction_set")
        names=[o["id"] for o in spec["options"]]
        value=names[best]
        if spec["type"]=="noul":value=bool(best)
        if spec["type"]=="score":value=sum(a*b for a,b in zip(p,spec["values"]))
        result={"type":spec["type"],"value":None if reasons else value,
                "suggested_value":value,"probabilities":dict(zip(names,p)),"logits":logits,
                "confidence":p[best],"abstained":bool(reasons),"abstain_reasons":reasons,
                "prediction_set":None if indices is None else [names[i] for i in indices],
                "temperature_used":temp,"calibration_language":language,
                "calibrated":self.calibration is not None,"smoke_only":self.manifest.get("smoke_only",False)}
        if spec["type"]=="noul":result["noul"]=p[1]
        if spec["type"]=="score":result["expected_value"]=value
        return result

    def decide(self,state,questions,min_confidence=0.0,language=None):
        if not isinstance(questions,dict) or not 1<=len(questions)<=64:
            raise ValueError("questions must map 1..64 names to question specifications")
        ctx=self.prepare(state)
        return {"answers":{name:self.score_prepared(ctx,q,min_confidence,language) for name,q in questions.items()},
                "model_sha256":self.manifest["model_sha256"]}

    def logits_row(self,row):
        ctx=self.prepare(row["state"])
        return self.score_prepared(ctx,row)

    def clear_cache(self):
        self.native.clear_cache(self.handle)

    @staticmethod
    def select_action(probabilities,cost_matrix,abstain_cost=None):
        return bayes_action(probabilities,cost_matrix,abstain_cost)
