import json
import math
from pathlib import Path
import subprocess
import sys
import struct
import torch
import pytest
from nanodecision2.config import ModelConfig
from nanodecision2.model import DecisionModel
from nanodecision2.export import write_native,export_bundle
from nanodecision2 import _native
from nanodecision2.data import collate,prepare_cache,CacheDataset
from nanodecision2.schema import normalize_question,gold_target,TextCodec
from nanodecision2.probability import softmax,metrics,fit_temperature,conformal_quantile,prediction_set,bayes_action
from nanodecision2.losses import proper_loss

@pytest.fixture
def model():
    torch.set_num_threads(1);torch.manual_seed(17)
    return DecisionModel(ModelConfig.smoke()).eval()

def example(c=3,length=13):
    return {"id":"row","group_id":"group","state_ids":[2]+list(range(5,5+length))+[1],
        "pair_ids":[[2,5+i,8+i,10+i,1] for i in range(c)],"type":"choice",
        "target_probs":[float(i==0) for i in range(c)],"option_ids":[str(i) for i in range(c)],"values":None,"weight":1.0}

@pytest.mark.parametrize("length",[1,9,65,137,255])
def test_native_encoder_matches(model,tmp_path,length):
    ids=[2]+[5+i%240 for i in range(length)]+[1];path=tmp_path/"m.nd2";write_native(model,path,False)
    engine=_native.load(str(path));actual=torch.tensor(_native.encode_ids(engine,ids)).reshape(len(ids),-1)
    expected=model.encoder(torch.tensor([ids]),torch.ones(1,len(ids),dtype=torch.bool))[0]
    assert torch.allclose(expected,actual,atol=5e-6,rtol=2e-5)

@pytest.mark.parametrize("count",[2,3,20,64,256])
def test_native_full_head_matches(model,tmp_path,count):
    r=example(count);r["pair_ids"]=[[2,5+i%240,7+(i*3)%240,1] for i in range(count)]
    b=collate([r]);z=model(**{k:b[k] for k in ("state_ids","state_mask","pair_ids","pair_mask","option_mask","state_index")})[0]
    path=tmp_path/"m.nd2";write_native(model,path,False);native=_native.load(str(path))
    actual=torch.tensor(_native.score(native,_native.prepare(native,r["state_ids"]),r["pair_ids"]))
    assert torch.allclose(z,actual,atol=3e-6,rtol=2e-5)


def test_quantized_matches_dequantized_reference(model,tmp_path):
    # Compare native to the exact dequantized weights, not to original FP32.
    clone=DecisionModel(model.cfg);state={}
    for k,v in model.state_dict().items():
        if v.ndim==2:
            scales=v.abs().amax(1).div(127).clamp_min(1e-12)
            state[k]=(v/scales[:,None]).round().clamp(-127,127)*scales[:,None]
        else:state[k]=v
    clone.load_state_dict(state);clone.eval();r=example();b=collate([r])
    z=clone(**{k:b[k] for k in ("state_ids","state_mask","pair_ids","pair_mask","option_mask","state_index")})[0]
    path=tmp_path/"q.nd2";write_native(model,path,True);native=_native.load(str(path))
    actual=torch.tensor(_native.score(native,_native.prepare(native,r["state_ids"]),r["pair_ids"]))
    assert torch.allclose(z,actual,atol=3e-6,rtol=2e-5)


def test_permutation(model,tmp_path):
    r=example();path=tmp_path/"m.nd2";write_native(model,path,False);native=_native.load(str(path));ctx=_native.prepare(native,r["state_ids"])
    first=_native.score(native,ctx,r["pair_ids"]);second=_native.score(native,ctx,list(reversed(r["pair_ids"])))
    assert first==pytest.approx(list(reversed(second)),abs=3e-7)


def test_state_sharing(model):
    rows=[example(),example()];rows[1]["pair_ids"][0]=[2,100,9,1]
    b=collate(rows);assert b["state_ids"].shape[0]==1
    together=model(**{k:b[k] for k in ("state_ids","state_mask","pair_ids","pair_mask","option_mask","state_index")})
    for i,r in enumerate(rows):
        one=collate([r]);z=model(**{k:one[k] for k in ("state_ids","state_mask","pair_ids","pair_mask","option_mask","state_index")})[0]
        assert torch.allclose(z,together[i],atol=1e-6)

@pytest.mark.parametrize("kind",["choice","noul","score"])
def test_schema_types(kind):
    if kind=="choice":row={"type":kind,"question":"Select","criteria":{"a":"A","b":"B"},"target":"a"}
    elif kind=="noul":row={"type":kind,"question":"True?","target":False}
    else:row={"type":kind,"question":"Level?","levels":[{"value":0,"description":"low"},{"value":10,"description":"high"}],"target":10}
    spec=normalize_question(row);assert sum(gold_target(row,spec))==1


def test_reject_ambiguous_or_invalid_targets():
    row={"type":"noul","question":"True?","target":0.5}
    with pytest.raises(ValueError):gold_target(row,normalize_question(row))
    with pytest.raises(ValueError):normalize_question({"type":"score","question":"q","levels":[{"value":2,"description":"a"},{"value":1,"description":"b"}]})
    with pytest.raises(ValueError):normalize_question({"question":"q","choices":[{"id":"a","description":"same"},{"id":"b","description":"same"}]})


def test_no_silent_truncation():
    with pytest.raises(ValueError):TextCodec(smoke=True).encode("abcd",3)


def test_opaque_ids_do_not_enter_model():
    a={"question":"Select","criteria":{"id1":"A","id2":"B"}}
    b={"question":"Select","criteria":{"different":"A","opaque":"B"}}
    codec=TextCodec(smoke=True)
    assert codec.pairs(normalize_question(a),256)==codec.pairs(normalize_question(b),256)


def test_proper_scores_minimized_at_truth():
    q=torch.tensor([[0.2,0.5,0.3]],dtype=torch.float64)
    mask=torch.ones_like(q,dtype=torch.bool);values=torch.tensor([[0.,1.,4.]])
    true_loss,_=proper_loss(q.log(),q,mask,values,torch.tensor([True]))
    for p in ([.1,.5,.4],[.3,.4,.3],[.6,.2,.2]):
        loss,_=proper_loss(torch.tensor([p]).log(),q,mask,values,torch.tensor([True]))
        assert loss>true_loss
    z=q.log().float().requires_grad_();loss,_=proper_loss(z,q.float(),mask,values,torch.tensor([True]));loss.sum().backward()
    assert z.grad.abs().max()<1e-6


def test_ordinal_distance_matters():
    q=[1.,0.,0.];close=metrics([.6,.4,0.],q,[0,1,2]);far=metrics([.6,0.,.4],q,[0,1,2])
    assert close["normalized_crps"]<far["normalized_crps"]


def test_temperature_never_worsens_its_fit_set():
    rows=[{"logits":[5.,-5.],"target_probs":[1.,0.]}]*7+[{"logits":[5.,-5.],"target_probs":[0.,1.]}]*3
    fit=fit_temperature(rows);assert fit["after_nll"]<=fit["before_nll"];assert fit["temperature"]>1


def test_conformal_off_by_one_and_small_sample():
    assert conformal_quantile([.1,.2,.3,.4],.2)==.4
    assert conformal_quantile([.1,.2,.3,.4],.05)==1.0
    assert prediction_set([.6,.4],.5)==[0]
    assert prediction_set([.6,.4],1.0)==[0,1]


def test_cost_sensitive_action():
    assert bayes_action([.6,.4],[[0,100],[1,0]])["action"]==1
    assert bayes_action([.6,.4],[[0,100],[1,0]],.1)["action"] is None

@pytest.mark.parametrize("damage",["truncate","magic","dimension"])
def test_malformed_native_file(model,tmp_path,damage):
    path=tmp_path/"m.nd2";write_native(model,path,False);raw=bytearray(path.read_bytes())
    if damage=="truncate":raw=raw[:20]
    elif damage=="magic":raw[0]=0
    else:struct.pack_into("<I",raw,16,0)
    path.write_bytes(raw)
    with pytest.raises(RuntimeError):_native.load(str(path))


def test_foreign_context_rejected(model,tmp_path):
    path=tmp_path/"m.nd2";write_native(model,path,False);a=_native.load(str(path));b=_native.load(str(path))
    ctx=_native.prepare(a,[2,5,1])
    with pytest.raises(RuntimeError):_native.score(b,ctx,[[2,6,1],[2,7,1]])


def test_runtime_does_not_import_training_libraries(model,tmp_path):
    export_bundle(model,tmp_path/"bundle",quantize=True,smoke_only=True,max_pair_tokens=256)
    program='''import sys, importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','numpy','transformers','onnxruntime','datasets'}: raise RuntimeError('Blocked '+fullname)
sys.meta_path.insert(0,Deny())
from nanodecision2 import DecisionEngine
engine=DecisionEngine(sys.argv[1],require_calibration=False)
result=engine.decide('status ok',{'q':{'type':'noul','question':'Is it done?'}})
assert result['answers']['q']['smoke_only']
assert result['answers']['q']['abstained']
print('native-runtime-ok')
'''
    result=subprocess.run([sys.executable,"-c",program,str(tmp_path/"bundle")],capture_output=True,text=True,check=True)
    assert "native-runtime-ok" in result.stdout


def test_streaming_probability_diagnostics():
    from nanodecision2.probability import EvaluationAccumulator
    a=EvaluationAccumulator(qhat=.3)
    a.add([.8,.2],[1.,0.],group_id="g1")
    a.add([.9,.1],[0.,1.],group_id="g1")
    a.add([.3,.7],[0.,1.],group_id="g2")
    r=a.summary()
    assert r["accuracy"]==pytest.approx(2/3)
    assert r["conformal"]["group_simultaneous_coverage"]==.5
    assert r["conformal"]["row_coverage"]==pytest.approx(2/3)
    assert r["fixed_threshold_diagnostics"]["0.9"]["accepted_accuracy"]==0
