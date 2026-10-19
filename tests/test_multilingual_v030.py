import copy
import json
import math
from argparse import Namespace
import pytest
import torch
from nanodecision2.multilingual import (language_key,probability_bucket,served_temperature,
    validate_language_weights,reviewed_pair_units,bilingual_js,paired_batches)
from nanodecision2.adapters import nli_row,paws_row,massive_row,typed_rows,simple_source_rows,SCENARIOS_KO
from nanodecision2.benchmark import make_record,record_metrics,summarize,align_records,cluster_bootstrap_delta
from nanodecision2.adaptation import mask_tokens,tied_mask_loss,prepare_corpus,adapt
from nanodecision2.initialization import initialize
from nanodecision2.training import read_model,train
from nanodecision2.data import prepare_cache,CacheDataset
from nanodecision2.export import export_bundle
from nanodecision2 import DecisionEngine


def test_language_keys_and_table_fallback():
    assert language_key('ko_KR')=='ko'
    assert language_key(None)=='unknown'
    c={'temperature':2.,'temperatures_by_bucket':{'ko|choice|3-5':{'temperature':3.}}}
    assert served_temperature(c,'ko-KR','choice',4)==3
    assert served_temperature(c,'en','choice',4)==2
    assert served_temperature(None,'ko','noul',2)==1
    assert probability_bucket('ko','choice',60)=='ko|choice|21+'

@pytest.mark.parametrize('weights',[{'ko':0},{'ko':float('nan')},{'ko':1,'ko-KR':2},[]])
def test_language_weight_validation(weights):
    with pytest.raises(ValueError):validate_language_weights(weights)

def test_nli_neutral_not_false_and_feature_order():
    r=nli_row({'premise':'고양이가 있다.','hypothesis':'고양이가 검다.','label':0},'fixture',label_names=['neutral','entailment','contradiction'])[0]
    assert r['target']=='neutral' and len(r['criteria'])==3
    assert '결정할 수 없다' in r['criteria']['neutral']
    with pytest.raises(ValueError):nli_row({'premise':'a','hypothesis':'b','label':0},'fixture')

def test_official_test_and_parallel_grouping():
    r={'id':9,'sentence1':'가','sentence2':'나','label':0}
    a=paws_row(r,'test','ko')[0];b=paws_row(r,'test','en')[0]
    assert a['group_id']==b['group_id'] and a['partition']=='test'
    assert paws_row(r,'train','ko')[0]['label_origin']=='machine_translated_train'

def test_massive_all_candidates_and_not_automatically_reviewed():
    base={'id':37,'locale':'ko-KR','partition':'test','scenario':'weather','intent':'weather_query','utt':'내일 비 와?'}
    a=massive_row(base)[0];b=massive_row({**base,'locale':'en-US','utt':'Will it rain tomorrow?'})[0]
    assert len(a['criteria'])==18 and a['group_id']==b['group_id']
    assert a['equivalence_verified'] is False and a['partition']=='test'
    with pytest.raises(ValueError):massive_row(base,'intent',labels=['weather_query'])

def test_typed_teacher_mode_is_not_reference_label():
    raw={'id':'a','state':json.dumps({'input':'visible'}),'facts':{'secret_answer':'hidden'},
        'questions':json.dumps({'q':{'type':'noul','instructions':'Did it pass?'}}),
        'gold':json.dumps({'q':{'probabilities':{'false':.6,'true':.4},'label':True}})}
    row=typed_rows(raw,'test')[0]
    assert row['target_probs']==[.6,.4] and row['reference_option_id']=='true'
    assert row['criteria']=={'false':'false','true':'true'} and 'facts' not in row['state']
    rec=make_record(row,'fixture');rec.update(probabilities=[.7,.3],status='ok')
    m=record_metrics(rec);assert m['accuracy']==0 and m['agreement_to_target_mode']==1
    assert row['label_origin']=='teacher_distribution_synthetic'

def pair_rows():
    x={'group_id':'g','type':'choice','values':None,'option_ids':['a','b'],
       'target_probs':[1.,0.],'equivalence_id':'p','equivalence_verified':True,'language':'ko'}
    return [x,{**x,'language':'en'}]

def test_reviewed_pairs_require_matching_origins_and_labels():
    rows=pair_rows();assert reviewed_pair_units(rows)==[(0,1)]
    bad=copy.deepcopy(rows);bad[1]['group_id']='different'
    with pytest.raises(ValueError):reviewed_pair_units(bad)
    bad=copy.deepcopy(rows);bad[1]['target_probs']=[0.,1.]
    with pytest.raises(ValueError):reviewed_pair_units(bad)
    with pytest.raises(ValueError):reviewed_pair_units([{**x,'equivalence_verified':False} for x in rows])

def test_bilingual_js_aligns_permuted_candidates():
    z=torch.tensor([[2.,-1.],[-1.,2.]],requires_grad=True)
    loss,n=bilingual_js(z,pair_rows(),torch.tensor([[0,1],[1,0]]))
    assert n==1 and abs(float(loss.detach()))<1e-7
    loss2,n=bilingual_js(z,pair_rows(),torch.tensor([[0,1],[0,1]]))
    assert float(loss2.detach())>0;loss2.backward();assert torch.isfinite(z.grad).all()

def test_pair_batch_never_splits_pair():
    units=[(0,1),(2,),(3,4),(5,)]
    batches=list(paired_batches(units,[7,7,2,9,9,3],3,42))
    assert sorted(sum(batches,[]))==list(range(6))
    assert any(0 in b and 1 in b for b in batches) and any(3 in b and 4 in b for b in batches)


def benchmark_rows():
    result=[]
    for i in range(4):
        row={'id':str(i),'group_id':'g'+str(i//2),'state':str(i),'type':'noul','question':'Done?','target':True,'language':'ko'}
        rec=make_record(row,'fixture');rec.update(status='ok',probabilities=[.3,.7]);result.append(rec)
    return result

def test_benchmark_failure_accounting_and_input_identity():
    rows=benchmark_rows();rows[0].update(status='failed',error_kind='unsupported_input')
    report=summarize(rows);assert report['coverage']==.75 and report['requested']==4
    assert report['resolved_correct_fraction_of_all']==.75
    a=rows[1];b={**a,'input_sha256':'different'}
    with pytest.raises(ValueError,match='input_sha256'):align_records(a,b)

def test_cluster_bootstrap_identical_zero():
    rows=benchmark_rows();out=cluster_bootstrap_delta(rows,copy.deepcopy(rows),repeats=100)
    assert out['groups']==2 and out['deltas']['accuracy']['interval']==[0.,0.]
    assert out['deltas']['nll']['value']==0


def test_mlm_mask_and_chunked_loss_gradient():
    g=torch.Generator().manual_seed(2);ids=torch.tensor([[2,9,8,7,1,0],[2,6,5,1,0,0]])
    x,m,v=mask_tokens(ids,0,3,20,g)
    assert not m[:,0].any() and not m[0,4] and not m[1,3] and not m[~v].any()
    h=torch.randn(5,8,requires_grad=True);w=torch.randn(20,8,requires_grad=True);y=torch.tensor([1,5,7,9,10])
    actual=tied_mask_loss(h,y,w,2);expected=torch.nn.functional.cross_entropy(h@w.T,y)
    assert torch.allclose(actual,expected,atol=1e-6)
    ga=torch.autograd.grad(actual,(h,w),retain_graph=True);gb=torch.autograd.grad(expected,(h,w))
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(ga,gb))

def test_encoder_adaptation_export_stays_native_compatible(tmp_path):
    initialize(tmp_path/'init',smoke=True)
    corpus=[{'text':f'{lang} corpus example {part} {i}.','group_id':f'{part}-{lang}-{i}',
             'partition':part,'language':lang} for part in ['train','validation'] for lang in ['ko','en'] for i in range(2)]
    p=tmp_path/'corpus.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in corpus))
    meta=prepare_corpus(p,tmp_path/'corpus.sqlite',tmp_path/'init',32);assert meta['counts']['validation']>0
    a=Namespace(init=str(tmp_path/'init'),out=str(tmp_path/'adapt'),corpus=str(tmp_path/'corpus.sqlite'),threads=1,seed=42,
       device='cpu',epochs=1,batch_size=2,lr=.0001,mask_chunk=4,mask_token_id=None,max_steps=1,bf16=False,gradient_checkpointing=False)
    report=adapt(a);assert report['new_runtime_parameters']==0
    old=read_model(tmp_path/'init');model=read_model(tmp_path/'adapt'/'best')
    assert old.state_dict().keys()==model.state_dict().keys()
    assert not torch.equal(old.encoder.embedding.weight,model.encoder.embedding.weight)
    src=tmp_path/'adapt'/'best';info=json.loads((src/'init.json').read_text())
    export_bundle(model,tmp_path/'bundle',src,True,True,128,128,info)
    engine=DecisionEngine(tmp_path/'bundle');answer=engine.decide('test',{'q':{'type':'noul','question':'Done?'}},language='ko')
    assert answer['answers']['q']['abstained']

def test_corpus_group_leak_rejected(tmp_path):
    initialize(tmp_path/'init',smoke=True)
    rows=[{'text':part,'group_id':'shared','partition':part,'language':'ko'} for part in ['train','validation']]
    p=tmp_path/'x.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='leaks'):prepare_corpus(p,tmp_path/'c.db',tmp_path/'init',32)


def test_language_calibration_runtime_and_pair_training(tmp_path):
    from test_training import make_rows,arguments
    from nanodecision2.evaluation import evaluate_cache,calibrate_bundle
    initialize(tmp_path/'init',smoke=True)
    raw=make_rows()
    for i,r in enumerate(raw):r['language']='ko' if i%2==0 else 'en'
    # Exactly two reviewed equivalent views per source, labels and candidate IDs unchanged.
    pairs=[]
    for r in raw[:4]:
        if r['partition']!='train':continue
        r.update(equivalence_id=r['id'],equivalence_verified=True,language='ko')
        b={**r,'id':r['id']+'-en','language':'en','state':r['state']+' English equivalent.'};pairs.append(b)
    raw.extend(pairs)
    path=tmp_path/'rows.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
    prepare_cache([path],tmp_path/'cache',tmp_path/'init',128,128)
    ds=CacheDataset(tmp_path/'cache','train');assert len(reviewed_pair_units(ds))==24
    a=arguments(tmp_path,'run');a.epochs=1;a.batch_size=4;a.pair_consistency=.05;a.language_weights={'ko':2.,'en':1.};a.selection_metric='macro_language_nll'
    result=train(a);assert result['reviewed_pair_comparisons']>=4
    assert result['language_exposure']['ko']['effective_weight']==2*result['language_exposure']['ko']['rows']
    src=tmp_path/'run'/'best';model=read_model(src);info=json.loads((src/'init.json').read_text())
    export_bundle(model,tmp_path/'bundle',src,True,True,128,128,info)
    for part in ('temperature','conformal'):evaluate_cache(tmp_path/'bundle',tmp_path/'cache',part,tmp_path/(part+'.jsonl'))
    cal=calibrate_bundle(tmp_path/'bundle',tmp_path/'temperature.jsonl',tmp_path/'conformal.jsonl',language_calibration=True,min_bucket_groups=2)
    assert 'temperatures_by_bucket' in cal
    engine=DecisionEngine(tmp_path/'bundle')
    a=engine.decide('test',{'q':{'type':'noul','question':'Done?'}},language='ko')['answers']['q']
    assert a['calibration_language']=='ko'


def test_three_model_report_common_intersection(tmp_path):
    from scripts.compare_predictions import compare
    paths=[]
    for name in ('nd2','julia','laya'):
        rows=benchmark_rows()
        if name=='laya':rows[0].update(status='failed',error_kind='unsupported_input')
        path=tmp_path/(name+'.jsonl');path.write_text(''.join(json.dumps(x)+'\n' for x in rows));paths.append(str(path))
    result=compare(paths,100,42)
    assert len(result['pairwise'])==3
    assert result['by_language']['ko']['nd2']['requested']==3


def test_reconcile_exact_states_without_moving_official_test(tmp_path):
    from scripts.reconcile_groups import reconcile
    rows=[{'id':str(i),'group_id':str(i),'state':'same','source_split':'train','partition':'train'} for i in range(2)]
    p=tmp_path/'rows.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    report=reconcile([p],tmp_path/'ok.jsonl',True);assert report['groups']==1 and report['merged_group_links']==1
    rows[1].update(source_split='test',partition='test');p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='fixed partitions'):reconcile([p],tmp_path/'fail.jsonl',True)


def test_tokenizer_audit_uses_real_pair_format(tmp_path):
    from scripts.audit_tokenizer import audit
    initialize(tmp_path/'init',smoke=True)
    r={'id':'1','group_id':'g','state':'hello','type':'noul','question':'Done?','target':True,'language':'ko'}
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps(r)+'\n')
    report=audit(tmp_path/'init',path,max_state=8,max_pair=8)
    assert report['by_language']['ko']['state_overflows']==1
    assert report['by_language']['ko']['candidate_pair_overflows']==2


def test_laya_adapter_contract_and_truncation_guard(monkeypatch):
    import sys,types
    from scripts.evaluate_raw import laya_predictor,UnsupportedInput
    class Tok:
        mask_token='[MASK]';mask_token_id=3;cls_token_id=2;sep_token_id=1;pad_token_id=0
        def __init__(self):self.vocab={}
        def __call__(self,text,**kwargs):
            return {'input_ids':[self.vocab.setdefault(w,len(self.vocab)+4) for w in text.split()]}
    tok=Tok();common=types.ModuleType('laya.common')
    common.QTYPES={'choice':0,'score':1,'noul':2};common.temp_bucket=lambda qt,k:f'{qt}:{k}'
    common.serialize_state=lambda x:x if isinstance(x,str) else json.dumps(x,ensure_ascii=False)
    common.render_options=lambda q:[f'{k}: {v}' for k,v in q['crit'].items()]
    def builder(tok,state,q,max_len,head_len,return_stats=False):
        opts=[tok(' '+x)['input_ids'] for x in common.render_options(q)]
        ids=[2]+tok(q['t']+' question: '+q['ins'])['input_ids']+[1];markers=[]
        for o in opts:markers.append(len(ids));ids+=[3]+o
        ids+=[1]+tok(common.serialize_state(state))['input_ids']+[1]
        return ids[:max_len],markers,{'options_distinct':len(opts)}
    common.build_sequence=builder
    def collate(items,pad):
        item=items[0][0]
        return {'input_ids':torch.tensor([item['ids']]),'attention_mask':torch.ones(1,len(item['ids']),dtype=torch.bool),
         'marker_pos':torch.tensor([item['markers']]),'marker_mask':torch.ones(1,len(item['markers']),dtype=torch.bool),'qtype':torch.tensor([item['qtype']])}
    common.collate_items=collate;monkeypatch.setitem(sys.modules,'laya.common',common)
    agent=types.SimpleNamespace(tok=tok,device='cpu',temperature_by_options={},temperature=[1.,1.,2.],
      model=lambda *args:(torch.tensor([[0.,1.]]),None))
    row={'state':'a normal state','type':'noul','question':'Done?','criteria':{'false':'no','true':'yes'}}
    pred=laya_predictor(agent,100,80)(row);assert pred['strict_input_verified'] and pred['temperature_used']==2.
    with pytest.raises(UnsupportedInput):laya_predictor(agent,4,80)(row)
    with pytest.raises(UnsupportedInput):laya_predictor(agent,100,80)({**row,'state':'contains [MASK]'})


def test_local_public_adapter_cli_and_massive_archive(tmp_path):
    import subprocess,sys,tarfile,io
    from scripts.build_data import massive_archive
    raw={'id':1,'locale':'ko-KR','partition':'test','utt':'날씨 알려줘','scenario':'weather','intent':'weather_query'}
    data=(json.dumps(raw,ensure_ascii=False)+'\n').encode()
    tarpath=tmp_path/'massive.tar.gz'
    with tarfile.open(tarpath,'w:gz') as tar:
        member=tarfile.TarInfo('data/ko-KR.jsonl');member.size=len(data);tar.addfile(member,io.BytesIO(data))
    assert list(massive_archive(tarpath,'ko-KR','test'))==[raw]
    out=tmp_path/'converted.jsonl'
    subprocess.run([sys.executable,'scripts/build_data.py','--source','massive-scenario','--language','ko-KR','--split','test','--archive',str(tarpath),'--out',str(out)],check=True,capture_output=True)
    row=json.loads(out.read_text());assert len(row['criteria'])==18 and row['language']=='ko-KR'
    manifest=json.loads(out.with_suffix('.manifest.json').read_text());assert manifest['decision_rows']==1 and manifest['by_partition']=={'test':1}
