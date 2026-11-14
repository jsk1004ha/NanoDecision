"""Second-stage contracts; tiny fixtures are not real model quality evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from nanodecision2.batching import token_batches,padded_cost,distributed_rounds
from nanodecision2.modalities import field_features,structured_tokens,sensor_tokens
from nanodecision2.quality_gates import QualityReport,protected_slices
from nanodecision2.verified_tasks import examples,oracle,world,generate
from nanodecision2.extended_model import ExtendedDecisionModel,ExtensionConfig,auxiliary_loss
from nanodecision2.config import ModelConfig
from nanodecision2.schema import TextCodec


def test_token_budget_never_drops_or_splits_pairs():
    shapes=np.array([[10,2,8],[20,3,8],[11,2,9],[30,4,10],[5,2,4]])
    batches=list(token_batches(shapes,3,300,units=[(0,2),(1,),(3,),(4,)]))
    assert sorted(i for b in batches for i in b)==list(range(5))
    assert any(0 in b and 2 in b for b in batches)
    assert all(padded_cost(shapes,b)<=300 for b in batches)
    assert batches==list(token_batches(shapes,3,300,units=[(0,2),(1,),(3,),(4,)]))
    rounds=list(distributed_rounds(batches,2))
    assert sorted(i for r in rounds for b in r for i in b)==list(range(5))
    with pytest.raises(ValueError):list(token_batches(shapes,2,10))


def test_per_task_guard_cannot_hide_english_regression():
    def report(values):
        r=QualityReport()
        for task,z in zip(('a','b'),values):
            r.add({'type':'choice','option_ids':['a','b'],'target_probs':[1.,0.],'values':None,'task':task,'language':'en'},z)
        return r.summary()
    baseline=report([[1.,0.],[1.,0.]])
    candidate=report([[10.,0.],[0.,1.]])
    assert not protected_slices(candidate,baseline)['passed']
    assert protected_slices(baseline,baseline)['passed']


def test_structured_zero_null_missing_and_row_groups():
    assert len({tuple(field_features(x)) for x in (None,0,False,'')})==4
    a={'modality':'table','columns':[{'name':'name'},{'name':'value','unit':'C'}],
       'rows':[{'name':'A','value':0},{'name':'B'}]}
    r=structured_tokens(a)
    assert r['groups']==[0,0,1,1] and r['positions']==[0,0,0,0]
    assert r['features'][-1][0]==1
    a['ordered_rows']=True
    assert structured_tokens(a)['positions']==[1,1,2,2]
    with pytest.raises(ValueError):structured_tokens(a,max_cells=2)


def series():
    return {'modality':'timeseries','values':[[0.,1.],[2.,None],[4.,5.]],'sample_rate_hz':10.,
            'channel_names':['x','y'],'units':['m/s2','C'],'observed_mask':[[True,True],[True,False],[True,True]]}


def test_sensor_mask_absolute_values_time_and_shape():
    row=series();r=sensor_tokens(row,patch_size=2)
    assert len(r['features'])==4 and r['frames']==3 and r['observed_count']==5
    changed=copy.deepcopy(row);changed['values'][1][1]=float('nan')
    assert sensor_tokens(changed,2)==r  # declared missing, never treated as observed zero
    bad=copy.deepcopy(row);bad['observed_mask'][1][1]=True
    with pytest.raises(ValueError):sensor_tokens(bad,2)
    bad=copy.deepcopy(row);bad['timestamps']=[0,1,2]
    with pytest.raises(ValueError):sensor_tokens(bad,2)
    bad=copy.deepcopy(row);bad['prediction_time']=.1
    with pytest.raises(ValueError):sensor_tokens(bad,2)
    shifted=copy.deepcopy(row);shifted['values'][0][0]=100.
    assert sensor_tokens(shifted,2)['features']!=r['features']


@pytest.mark.parametrize('mutation',['jagged','mask','units','file'])
def test_bad_sensor_contract(mutation):
    r=series()
    if mutation=='jagged':r['values'][1].pop()
    elif mutation=='mask':r['observed_mask'][0][0]=1
    elif mutation=='units':r['units'].pop()
    else:r['values_ref']='/etc/passwd'
    with pytest.raises(ValueError):sensor_tokens(r)


def extended(**kwargs):
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=ExtensionConfig(block_tokens=48,max_pair_tokens=256,metadata_tokens=256,max_fused_tokens=512,
                        structured=True,sensors=True,token_interaction=True,evidence=True,relation_classes=3,**kwargs)
    return ExtendedDecisionModel(ModelConfig.smoke(),cfg)


def test_extended_modalities_gradients_no_abstention_gate():
    model=extended();codec=TextCodec(smoke=True)
    question={'type':'noul','question':'Is the condition satisfied?'}
    inputs=['a long enough text '*8,{'modality':'records','fields':[{'name':'temperature','value':35,'unit':'C'}]},series()]
    for state in inputs:
        model.zero_grad(set_to_none=True)
        z,aux,details=model(state,question,codec,True)
        assert z.shape==(2,) and torch.isfinite(z).all()
        (z.sum()+auxiliary_loss(aux,{'relation_id':1})).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if isinstance(state,str):assert details['context_mode']=='all_blocks' and details['encoded_tokens']>48
    assert any(p.grad is not None for p in model.sensor_encoder.parameters())
    with pytest.raises(ValueError,match='no legacy native'):
        from nanodecision2.export import write_native
        write_native(model,'unused.nd2')


def test_table_column_row_permutation_invariance_but_preserves_association():
    model=extended().eval();codec=TextCodec(smoke=True)
    a={'modality':'table','columns':[{'name':'name'},{'name':'value'}],
       'rows':[{'name':'A','value':1},{'name':'B','value':10}]}
    b=copy.deepcopy(a);b['columns'].reverse();b['rows'].reverse()
    c=copy.deepcopy(a);c['rows'][0]['value']=10;c['rows'][1]['value']=1
    q={'type':'noul','question':'Does A have the higher value?'}
    with torch.no_grad():za=model(a,q,codec);zb=model(b,q,codec);zc=model(c,q,codec)
    assert torch.allclose(za,zb,atol=2e-6)
    assert not torch.equal(za,zc)


def test_generator_has_eight_language_combinations_and_label_oracles(tmp_path):
    rows=list(examples(0,42));assert len(rows)==60
    assert len({(r['state_language'],r['question_language'],r['candidate_languages']) for r in rows if not r['code_switch']})==8
    facts=world(0,42)
    assert all(r['target']==oracle(facts,r['task']) for r in rows)
    assert len({r['group_id'] for r in rows})==1 and len({r['partition'] for r in rows})==1
    report=generate(tmp_path/'data.jsonl',3)
    assert report['source_worlds']==3 and report['decision_rows']==180
    again=generate(tmp_path/'data2.jsonl',3)
    assert report['sha256']==again['sha256']
    with pytest.raises(FileExistsError):generate(tmp_path/'data.jsonl',3)


def test_fewshot_has_separate_worlds_and_query_target_is_not_injected():
    row=next(examples(0,42,fewshot=4));assert len(row['state']['support_examples'])==4
    assert 'target' not in row['state'] and 'query' in row['state']
    assert all(s['state']!=row['state']['query'] for s in row['state']['support_examples'])


def test_jevbench_mapping_and_target_never_read(monkeypatch):
    from nanodecision2.jevbench_adapter import map_question,NanoDecisionAdapter
    import types
    @__import__('dataclasses').dataclass
    class Result:
        adapter:str;ok:bool;probs:dict=None;probs_source:str='';model:str='';status:int=None;error:str=None
        latency_s:float=0.;usage:dict=None;raw:dict=None;request_body:dict=None
    monkeypatch.setitem(sys.modules,'jevbench.adapters.base',types.SimpleNamespace(DecisionResult=Result))
    class Engine:
        manifest={'model_sha256':'fixture'}
        def decide(self,state,questions):
            assert state=='input only' and 'expected' not in questions['decision']
            return {'answers':{'decision':{'probabilities':{'false':.4,'true':.6},'logits':[0.,.4]}}}
    class Task:
        state='input only';labels=['yes','no'];question={'type':'noul','instructions':'True?'}
        @property
        def expected(self):raise AssertionError('Gold leakage')
    result=NanoDecisionAdapter(engine=Engine()).run(Task())
    assert result.ok and result.probs=={'yes':.6,'no':.4}
    q,m=map_question({'type':'score','instructions':'Rate','criteria':['low','high']},['0','1'])
    assert m=={'0':'0','1':'1'}
    with pytest.raises(ValueError):map_question({'type':'choice','criteria':{'a':'A'}},['a','b'])


def create_cache(root):
    from nanodecision2.initialization import initialize
    from nanodecision2.data import prepare_cache
    initialize(root/'init',smoke=True)
    rows=[]
    for part,n in [('train',13),('validation',8),('temperature',4),('conformal',0),('test',4)]:
        for i in range(n):rows.append(dict(id=f'{part}{i}',group_id=f'{part}{i}',partition=part,state=f'{part} record {i}. Tests '+('passed' if i%2 else 'failed'),type='noul',question='Tests passed?',target=bool(i%2),language='en' if i%2 else 'ko',task='status'))
    source=root/'rows.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    prepare_cache([source],root/'cache',root/'init',128,128)
    return source


def test_parallel_preparation_is_byte_identical_and_group_audit_stays_single_writer(tmp_path):
    from nanodecision2.data import prepare_cache
    source=create_cache(tmp_path)
    serial=json.loads((tmp_path/'cache'/'manifest.json').read_text())
    parallel=prepare_cache([source],tmp_path/'parallel',tmp_path/'init',128,128,workers=2)
    assert serial==parallel


def test_scalable_resume_and_per_task_guards(tmp_path):
    from nanodecision2.scalable_training import parser,train
    from nanodecision2.training import read_model
    create_cache(tmp_path)
    def args(out):
        a=parser().parse_args(['--init',str(tmp_path/'init'),'--cache',str(tmp_path/'cache'),'--out',str(tmp_path/out),
            '--device','cpu','--batch-size','3','--accumulation','2','--threads','1','--eval-every','2','--save-every','1','--rdrop','.1'])
        a.protect_language=['en','ko'];return a
    full=train(args('full'));a=args('resumed');a.stop_after=1;train(a)
    a=args('resumed');a.resume=str(tmp_path/'resumed'/'last.pt');continued=train(a)
    x=read_model(tmp_path/'full'/'final').state_dict();y=read_model(tmp_path/'resumed'/'final').state_dict()
    assert all(torch.equal(x[k],y[k]) for k in x)
    assert full['global_rows_seen']==13 and continued['global_rows_seen']==13


def test_extended_train_save_reload_and_tamper(tmp_path):
    from nanodecision2.initialization import initialize
    from nanodecision2.extended_workflow import parser,train,ExtendedEngine
    initialize(tmp_path/'init',smoke=True)
    rows=[]
    for part in ('train','validation','temperature'):
        for i in range(4):
            s=series();s['values'][0][0]=i+{'train':0,'validation':100,'temperature':200}[part]
            rows.append({'id':f'{part}{i}','group_id':f'{part}{i}','partition':part,'state':s,
                'question_spec':{'type':'noul','question':'Is the first reading above 1?'},'language':'en','task':'threshold',
                'supervision':{'target':s['values'][0][0]>1,'relation_id':i%3}})
    p=tmp_path/'rows.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    a=parser().parse_args(['--init',str(tmp_path/'init'),'--input',str(p),'--out',str(tmp_path/'run'),'--sensors',
        '--relation-classes','3','--block-tokens','48','--max-pair','256','--metadata-tokens','256','--threads','1'])
    report=train(a);assert report['training_complete'] and not report['native_export_supported']
    from nanodecision2.extended_inference import calibrate
    cal=calibrate(tmp_path/'run'/'final',p,allow_smoke=True)
    assert cal['n']==4 and cal['method']=='temperature_only'
    engine=ExtendedEngine(tmp_path/'run'/'final',allow_smoke=True)
    answer=engine.decide(series(),{'q':{'type':'noul','question':'Is there an event?'}})['answers']['q']
    assert answer['value'] is not None and answer['context']['context_mode']=='timeseries' and answer['calibrated']
    (tmp_path/'run'/'final'/'extension.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):ExtendedEngine(tmp_path/'run'/'final',allow_smoke=True)


def test_all_blocks_native_matches_explicit_torch_feature_concatenation(tmp_path):
    from nanodecision2.initialization import initialize
    from nanodecision2.training import read_model
    from nanodecision2.export import export_bundle
    from nanodecision2.long_context import AllBlocksEngine
    from nanodecision2 import _native
    initialize(tmp_path/'init',smoke=True)
    model=read_model(tmp_path/'init').eval()
    export_bundle(model,tmp_path/'bundle',tmp_path/'init',False,True,128,256)
    engine=AllBlocksEngine(tmp_path/'bundle',block_tokens=32,max_tokens=512,allow_smoke=True)
    state='meaningful state '*8;question={'type':'noul','question':'Did it happen?'}
    result=engine.decide(state,{'q':question})['answers']['q']
    ids=engine.engine.codec.state(state,512);blocks=[ids[i:i+32] for i in range(0,len(ids),32)]
    from nanodecision2.schema import normalize_question
    pairs=engine.engine.codec.pairs(normalize_question(question),256)
    with torch.no_grad():
        context=torch.cat([model.encoder(torch.tensor([b]),torch.ones(1,len(b),dtype=torch.bool))[0] for b in blocks])
        candidates=torch.stack([model.head.candidate(model.encoder(torch.tensor([p]),torch.ones(1,len(p),dtype=torch.bool))[0].mean(0)) for p in pairs])
        z=model.head(model.head.key(context)[None],model.head.value(context)[None],torch.ones(1,len(context),dtype=torch.bool),candidates[None],torch.ones(1,2,dtype=torch.bool))[0]
    assert result['logits']==pytest.approx(z.tolist(),abs=3e-6)
    assert result['value'] is not None and result['encoded_tokens']==len(ids)
    assert not result['calibrated'] and result['context_mode']=='all_blocks'
    # A direct-context temperature is intentionally not accepted as all-block calibration.
    manifest=json.loads((tmp_path/'bundle'/'manifest.json').read_text())
    (tmp_path/'bundle'/'calibration.json').write_text(json.dumps({'model_sha256':manifest['model_sha256'],'temperature':5.}))
    again=AllBlocksEngine(tmp_path/'bundle',32,512,True).decide(state,{'q':question})['answers']['q']
    assert again['temperature_used']==1.
    with pytest.raises(RuntimeError):_native.prepare_many(engine.engine.handle,[[2]*513])


def test_extended_text_auxiliary_supervision_bounds_and_missing_targets():
    model=extended().eval();codec=TextCodec(smoke=True)
    z,aux,_=model('some text',{'type':'noul','question':'True?'},codec,True)
    assert auxiliary_loss(aux,{}) is None
    loss=auxiliary_loss(aux,{'evidence_token_span':[1,3],'relation_id':0});loss.backward()
    assert model.evidence_start.weight.grad is not None
    with pytest.raises(ValueError):auxiliary_loss(aux,{'evidence_token_span':[999,1000]})


def test_generator_no_duplicate_input_question_semantics_across_origins():
    seen={}
    from nanodecision2.schema import state_text,normalize_question
    for index in range(20):
        for r in examples(index,42):
            key=state_text(r['state'])
            assert seen.setdefault(key,r['group_id'])==r['group_id']
    rows=list(examples(3,42))
    keys=[json.dumps({'state':r['state'],'spec':normalize_question(r)},sort_keys=True,ensure_ascii=False) for r in rows]
    assert len(keys)==len(set(keys))


def test_batch_probe_reports_measured_cpu_without_claiming_cuda(tmp_path):
    from nanodecision2.batch_probe import probe
    create_cache(tmp_path)
    torch.set_num_threads(1)
    report=probe(tmp_path/'init',tmp_path/'cache','cpu',(1,2),1)
    assert report['device']=='cpu' and len(report['results'])==2
    assert all(r['status']=='ok' and r['seconds_per_microbatch']>0 and r['peak_allocated'] is None for r in report['results'])


def test_parallel_source_probe_resolves_in_order_and_limits_shared_downloads(tmp_path,monkeypatch):
    from nanodecision2 import source_preflight
    import time,threading
    mutex=threading.Lock();active=[0];peak=[0];calls=[]
    class Reader:
        def __init__(self,path):pass
        def resolve(self,spec):return {'repo':'r','data_revision':'a'*40,'files':['same.jsonl']}
        def shards(self,spec):
            with mutex:active[0]+=1;peak[0]=max(peak[0],active[0]);calls.append(spec['id'])
            time.sleep(.005)
            with mutex:active[0]-=1
            yield 0,tmp_path/'sample.jsonl',{'file':'same.jsonl'}
    (tmp_path/'sample.jsonl').write_text(json.dumps({'premise':'A','hypothesis':'B','label':'neutral'})+'\n')
    monkeypatch.setattr(source_preflight,'PublicReader',Reader)
    specs=[{'id':str(i),'adapter':'nli','language':'en','split':'train'} for i in range(6)]
    r=source_preflight.preflight({'sources':specs},tmp_path/'out',workers=4)
    assert [x['id'] for x in r['sources']]==list(map(str,range(6)))
    assert len(calls)==6 and peak[0]==1 and all(x['status']=='ok' for x in r['sources'])


def test_two_process_ddp_global_weight_normalization(tmp_path):
    import os
    worker=Path(__file__).with_name('ddp_gradient_worker.py')
    output=tmp_path/'gradient.json'
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(worker),str(output)],
                   check=True,capture_output=True,text=True,timeout=40,
                   env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    result=json.loads(output.read_text())
    assert result['world_size']==2 and result['global_weight']==15 and result['max_abs_gradient_error']<=1e-12


def test_profiles_are_explicit_and_cannot_upgrade_smoke_position_limit():
    from nanodecision2.autopipeline import smoke_recipe
    from nanodecision2.training_profiles import apply_profile
    r=smoke_recipe();p=apply_profile(r,'batch128')
    assert p['training']['batch_size']*p['training']['accumulation']==128
    assert p['training']['trainer']=='scalable' and 'training_profile' not in r
    with pytest.raises(ValueError):apply_profile(r,'context8k')
