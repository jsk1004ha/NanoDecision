import json
from pathlib import Path
import copy
import torch
import pytest
from argparse import Namespace
from nanodecision2.initialization import initialize
from nanodecision2.data import prepare_cache,CacheDataset
from nanodecision2.training import train,read_model
from nanodecision2.export import export_bundle
from nanodecision2.evaluation import evaluate_cache,calibrate_bundle
from nanodecision2 import DecisionEngine


def make_rows():
    rows=[]
    sizes={"train":24,"validation":8,"temperature":8,"conformal":12,"test":8}
    for part,n in sizes.items():
        for i in range(n):
            rows.append({"id":f"{part}-{i}","group_id":f"group-{part}-{i}","partition":part,
                "state":f"Case {part} {i}. "+("Tests passed." if i%2 else "Tests failed."),
                "type":"noul","question":"Did the tests pass?","target":bool(i%2)})
    return rows


@pytest.fixture
def project(tmp_path):
    initialize(tmp_path/"init",smoke=True)
    path=tmp_path/"rows.jsonl";path.write_text("".join(json.dumps(r)+"\n" for r in make_rows()))
    prepare_cache([path],tmp_path/"cache",tmp_path/"init",max_state=128,max_pair=128)
    return tmp_path


def arguments(root,out):
    return Namespace(init=str(root/"init"),cache=str(root/"cache"),out=str(root/out),device="cpu",epochs=2,
        batch_size=3,accumulation=2,encoder_lr=.001,head_lr=.002,layer_decay=.9,weight_decay=.01,warmup_ratio=.05,
        brier_weight=.2,rps_weight=.2,eval_every=2,save_every=2,log_every=2,patience=20,seed=77,threads=1,
        bf16=False,gradient_checkpointing=False,resume=None,stop_after=0)


def test_full_training_native_calibration(project):
    args=arguments(project,"run");report=train(args)
    model=read_model(project/"run"/"best")
    info=json.loads((project/"run"/"best"/"init.json").read_text())
    export_bundle(model,project/"bundle",project/"run"/"best",True,True,128,128,info)
    for part in ("temperature","conformal","test"):
        evaluate_cache(project/"bundle",project/"cache",part,project/(part+".jsonl"))
    calibrated=calibrate_bundle(project/"bundle",project/"temperature.jsonl",project/"conformal.jsonl")
    assert calibrated["conformal_groups"]==12
    engine=DecisionEngine(project/"bundle")
    result=engine.decide("Tests passed",{"done":{"type":"noul","question":"Did the tests pass?"}})
    assert result["answers"]["done"]["calibrated"]
    assert report["optimizer_steps_total"]==8


def test_exact_resume_matches_uninterrupted(project):
    full=arguments(project,"full");train(full)
    partial=arguments(project,"resumed");partial.stop_after=2;train(partial)
    resumed=arguments(project,"resumed");resumed.resume=str(project/"resumed"/"last.pt");train(resumed)
    a=read_model(project/"full"/"final");b=read_model(project/"resumed"/"final")
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


def test_checkpointing_matches_forward(project):
    args=arguments(project,"checkpointed");args.gradient_checkpointing=True;args.epochs=1
    assert train(args)["optimizer_steps_total"]==4


def test_group_leakage_rejected(tmp_path):
    initialize(tmp_path/"init",smoke=True);rows=make_rows()[:2]
    rows[1]["group_id"]=rows[0]["group_id"];rows[1]["partition"]="test"
    p=tmp_path/"rows.jsonl";p.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError,match="two partitions"):prepare_cache([p],tmp_path/"cache",tmp_path/"init",128,128)


def test_same_state_different_group_rejected(tmp_path):
    initialize(tmp_path/"init",smoke=True);rows=make_rows()[:2];rows[1]["state"]=rows[0]["state"]
    p=tmp_path/"rows.jsonl";p.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError,match="different group_id"):prepare_cache([p],tmp_path/"cache",tmp_path/"init",128,128)


def test_cache_tamper_rejected(project):
    with open(project/"cache"/"train.jsonl","ab") as f:f.write(b"\n")
    with pytest.raises(ValueError,match="changed"):CacheDataset(project/"cache","train").verify()


def test_score_soft_label_rules():
    from nanodecision2.schema import normalize_question,gold_target
    row={"type":"score","question":"How much?","criteria":["low","mid","high"],"target":.5}
    with pytest.raises(ValueError):gold_target(row,normalize_question(row))
    row["target_probs"]=[.5,.5,0]
    assert gold_target(row,normalize_question(row))==[.5,.5,0]
