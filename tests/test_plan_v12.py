"""Acceptance tests for the v1.2 foundation; fixtures are not quality benchmarks."""
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import gzip
import json
import pickle
import sys
from types import SimpleNamespace

import pytest
import torch

from nanodecision2.decision_contract import decision_result, calibration_language
from nanodecision2.schema import normalize_question
from nanodecision2.source_contracts import select_files, metadata_files, delimited_rows, SourceContractError
from nanodecision2.auto_sources import PublicReader, raw_rows, convert
from nanodecision2.autopipeline import default_recipe, smoke_recipe
from nanodecision2.initialization import initialize
from nanodecision2.training import read_model
from nanodecision2.export import export_bundle
from nanodecision2.runtime import DecisionEngine


@pytest.fixture
def bundle(tmp_path):
    initialize(tmp_path/'init', smoke=True)
    export_bundle(read_model(tmp_path/'init'), tmp_path/'bundle', tmp_path/'init', True, True, 384, 256)
    return tmp_path/'bundle'


def calibration(bundle, **extra):
    manifest = json.loads((bundle/'manifest.json').read_text())
    result = dict(temperature=1., model_sha256=manifest['model_sha256'], **extra)
    (bundle/'calibration.json').write_text(json.dumps(result))


@pytest.mark.parametrize('p', [[.34,.33,.33], [1/3]*3, [.01,.01,.98]])
def test_choice_always_returns_best_candidate_and_ties_are_id_based(p):
    spec=normalize_question({'question':'Choose', 'criteria':{'z':'Z','a':'A','m':'M'}})
    a=decision_result(spec,p)
    assert a['value'] is not None and not a['abstained']
    b=decision_result({**spec,'options':list(reversed(spec['options']))},list(reversed(p)))
    assert a['selected_option_id']==b['selected_option_id']
    assert a['probabilities']==b['probabilities']
    if len(set(p))==1:assert a['value']=='a'


def test_noul_named_mapping_score_mode_and_expectation():
    spec={'type':'noul','options':[{'id':'true'},{'id':'false'}], 'values':None}
    result=decision_result(spec,[.8,.2]);assert result['value'] is True and result['noul']==.8
    spec=normalize_question({'type':'score','question':'Level?', 'levels':[
        {'value':0,'description':'Low'},{'value':10,'description':'High'}]})
    result=decision_result(spec,[.4,.6])
    assert result['value']==6 and result['expected_value']==6 and result['mode_value']==10
    assert result['top1_probability']==.6


@pytest.mark.parametrize('p',[[float('nan'),.4],[-.1,1.1],[.3,.3]])
def test_invalid_distributions_are_errors_not_predictions(p):
    with pytest.raises(ValueError):decision_result(normalize_question({'type':'noul','question':'Q?'}),p)


def test_raw_smoke_requires_explicit_permission_and_returns_a_decision(bundle):
    with pytest.raises(ValueError,match='allow_smoke'):DecisionEngine(bundle)
    answer=DecisionEngine(bundle,allow_smoke=True).decide('state',{'q':{'type':'noul','question':'Done?'}})['answers']['q']
    assert answer['value'] is not None and not answer['abstained']
    assert answer['calibrated'] is False and answer['status']=='test_only'
    assert answer['calibration_status']=='raw'


def test_old_qhat_and_high_threshold_do_not_erase_default_decision(bundle):
    calibration(bundle,qhat=1.)
    engine=DecisionEngine(bundle,allow_smoke=True)
    answer=engine.decide('state',{'q':{'type':'noul','question':'Done?'}},min_confidence=1.)['answers']['q']
    assert answer['value'] is not None and not answer['abstained']
    assert len(answer['prediction_set'])==2 and answer['policy_result']['review_recommended']
    old=DecisionEngine(bundle,allow_smoke=True,legacy_policy=True)
    old_answer=old.decide('state',{'q':{'type':'noul','question':'Done?'}})['answers']['q']
    assert old_answer['value'] is None and old_answer['suggested_value'] is not None
    assert 'non_singleton_prediction_set' in old_answer['abstain_reasons']


def test_hash_is_checked_even_without_calibration(bundle):
    with (bundle/'model.nd2').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='hash'):DecisionEngine(bundle,allow_smoke=True)


@pytest.mark.parametrize('extra',[{'temperature':0.},{'qhat':float('nan')}])
def test_invalid_calibration_is_technical_error(bundle,extra):
    calibration(bundle)
    p=bundle/'calibration.json';data=json.loads(p.read_text());data.update(extra);p.write_text(json.dumps(data))
    with pytest.raises(ValueError):DecisionEngine(bundle,allow_smoke=True)


def test_component_language_and_question_override():
    assert calibration_language({'language':'ko'},'en')=='ko'
    assert calibration_language({},'en',{'state_language':'ko','question_language':'en'})=='mixed'
    assert calibration_language({'question_language':'ko'},'en',{'state_language':'ko','question_language':'en'})=='ko'
    assert calibration_language({'candidate_languages':{'a':'en','b':'ko'}},'en')=='mixed'
    assert calibration_language({'code_switch':True},'en')=='mixed'


def test_mixed_temperature_runtime_and_cli_metadata(bundle,tmp_path,capsys):
    from nanodecision2.cli import main
    calibration(bundle,temperatures_by_bucket={'mixed|noul|2':{'temperature':2.5},'ko|noul|2':{'temperature':3.}})
    request={'state':'test','language':'en','state_language':'ko','question_language':'en',
        'questions':{'q':{'type':'noul','question':'Done?'}}}
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    main(['predict','--bundle',str(bundle),'--input',str(path),'--allow-smoke'])
    result=json.loads(capsys.readouterr().out)['answers']['q']
    assert result['temperature_used']==2.5 and result['calibration_language']=='mixed'
    request['questions']['q']['question_language']='ko'
    engine=DecisionEngine(bundle,allow_smoke=True)
    result=engine.decide(request['state'],request['questions'],language='en',
        language_metadata={'state_language':'ko','question_language':'en'})['answers']['q']
    assert result['temperature_used']==3.


def test_native_concurrent_requests_and_cache_eviction(bundle):
    from nanodecision2 import _native
    handle=_native.load(str(bundle/'model.nd2'))
    ids=[2,9,11,1];pairs=[[2,13,1],[2,17,18,1]]
    expected=_native.score(handle,_native.prepare(handle,ids),pairs)
    def run(index):
        if index%7==0:_native.clear_cache(handle)
        # Exercise independent contexts plus shared candidate-cache hits/misses.
        _native.score(handle,_native.prepare(handle,[2,4+index%240,1]),[[2,5+index%230,1],[2,6+index%230,1]])
        return _native.score(handle,_native.prepare(handle,ids),pairs)
    with ThreadPoolExecutor(max_workers=8) as pool:outputs=list(pool.map(run,range(320)))
    for actual in outputs:assert actual==pytest.approx(expected,abs=1e-6)


def test_literal_quotes_bom_crlf_trailing_empty_and_gzip_resume(tmp_path):
    raw='\ufeffa\tb\tc\r\n"unclosed\tvalue\t\r\n"quoted"\tother\tlast\r\n'
    path=tmp_path/'x.tsv.gz'
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:f.write(raw)
    spec={'id':'fixture','parser':{'quoting':'none','expected_columns':['a','b','c']}}
    rows=list(raw_rows(path,spec));assert len(rows)==2
    assert rows[0][1]=={'a':'"unclosed','b':'value','c':''}
    assert list(raw_rows(path,spec,start=1))==rows[1:]


def test_quoted_tab_and_newline_is_one_logical_record(tmp_path):
    path=tmp_path/'x.tsv';path.write_text('a\tb\n"one\ttwo\nthree"\t1\nplain\t2\n')
    rows=list(delimited_rows(path,{}))
    assert len(rows)==2 and rows[0][1]['a']=='one\ttwo\nthree'
    assert list(delimited_rows(path,{},start=1))==rows[1:]


@pytest.mark.parametrize('text,code',[
    ('a\ta\nx\ty\n','invalid_header'),('a\tb\nx\n','column_count'),
    ('a\tb\nx\ty\tz\n','column_count'),('a\tb\n"unclosed\ty\n','parse_error')])
def test_malformed_tsv_fails_without_dialect_guessing(tmp_path,text,code):
    path=tmp_path/'x.tsv';path.write_text(text)
    with pytest.raises(SourceContractError,match=code):list(delimited_rows(path,{}))


def test_delimiter_limit_is_restored_and_csv_defaults(tmp_path):
    path=tmp_path/'x.csv';path.write_text('a,b\nlongvalue,1\n')
    old=csv.field_size_limit()
    with pytest.raises(SourceContractError,match='parse_error'):
        list(delimited_rows(path,{'parser':{'field_size_limit':5}}))
    assert csv.field_size_limit()==old
    assert list(delimited_rows(path,{}))[0][1]['b']=='1'


def test_plain_text_layout_explicit_variant_and_ambiguity():
    files=['plain_text/train-00000.parquet','plain_text/test-00000.parquet']
    assert select_files(files,None,'train')==[files[0]]
    assert select_files(files,'plain_text','train')==[files[0]]
    assert select_files(['plain_text/train/0000.parquet'],'plain_text','train')==['plain_text/train/0000.parquet']
    assert select_files(['data/train_minedHN.jsonl'],None,'train',['train_minedHN.jsonl'],nested_explicit=True)==['data/train_minedHN.jsonl']
    with pytest.raises(SourceContractError,match='ambiguous'):
        select_files(['a/train_minedHN.jsonl','b/train_minedHN.jsonl'],None,'train',['train_minedHN.jsonl'],nested_explicit=True)
    with pytest.raises(SourceContractError,match='split_mismatch'):
        select_files(['test.jsonl'],None,'train',['test.jsonl'])


def test_metadata_configuration_and_no_unsafe_union():
    card={'configs':[{'config_name':'data','data_files':[{'split':'train','path':'custom/source.jsonl'}]}]}
    assert metadata_files(card,['custom/source.jsonl'],'data','train')==['custom/source.jsonl']
    with pytest.raises(SourceContractError,match='ambiguous_source'):
        select_files(['data/train-0.parquet','plain_text/train-0.parquet'],None,'train')


def test_missing_explicit_variant_never_uses_converted_parquet(monkeypatch,tmp_path):
    calls=[]
    class API:
        def dataset_info(self,repo,revision):calls.append(revision);return SimpleNamespace(sha='deadbeef',card_data={})
        def list_repo_files(self,*a,**kw):return ['data/train-0.parquet']
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(HfApi=API))
    with pytest.raises(SourceContractError,match='explicit_file_not_found'):
        PublicReader(tmp_path).resolve({'id':'wiki','repo':'example/wiki','split':'train','files':['train_minedHN.jsonl']})
    assert calls==['main']


def test_default_recipe_parser_and_no_conformal_training_contract():
    r=default_recipe();specs={s['id']:s for s in r['sources']}
    assert specs['snli']['config']=='plain_text'
    assert specs['kornli-snli']['parser']['quoting']=='none'
    assert specs['ko-wiki-pairs']['config']=='data'
    assert specs['ko-wiki-pairs']['variant']=='published_data_config'
    assert r['reserve_rows']['conformal']==0 and smoke_recipe()['reserve_rows']['conformal']==0


def test_source_preflight_is_before_model_and_stops_required_failure(monkeypatch,tmp_path):
    from nanodecision2 import source_preflight
    class Reader:
        def __init__(self,*a):pass
        def resolve(self,spec):raise SourceContractError('explicit_file_not_found','probe')
    monkeypatch.setattr(source_preflight,'PublicReader',Reader)
    with pytest.raises(RuntimeError,match='model download has not started'):
        source_preflight.preflight({'sources':[{'id':'x','adapter':'nli'}]},tmp_path)
    report=json.loads((tmp_path/'source_preflight.json').read_text())
    assert report['required_failures']==['x'] and not (tmp_path/'init').exists()


def test_temperature_only_bundle_keeps_decisions(bundle,tmp_path):
    from nanodecision2.evaluation import calibrate_bundle
    model_hash=json.loads((bundle/'manifest.json').read_text())['model_sha256']
    rows=[{'id':str(i),'group_id':str(i),'split':'temperature','model_sha256':model_hash,
        'logits':[.1,.9],'target_probs':[0.,1.],'type':'noul','language':'en'} for i in range(4)]
    path=tmp_path/'temp.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=calibrate_bundle(bundle,path)
    assert 'qhat' not in result and result['method']=='temperature_only'
    answer=DecisionEngine(bundle,allow_smoke=True).decide('x',{'q':{'type':'noul','question':'Q?'}})['answers']['q']
    assert answer['calibrated'] and answer['value'] is not None


def test_parallel_loader_and_resume_match_exact_cpu_training(tmp_path):
    from test_training import make_rows,arguments
    from nanodecision2.data import prepare_cache,CacheDataset
    from nanodecision2.loading import row_batches
    from nanodecision2.training import train
    initialize(tmp_path/'init',smoke=True)
    path=tmp_path/'rows.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in make_rows()))
    prepare_cache([path],tmp_path/'cache',tmp_path/'init',128,128)
    ds=CacheDataset(tmp_path/'cache','train')
    cloned=pickle.loads(pickle.dumps(ds));assert cloned[3]==ds[3];cloned.close()
    a=list(row_batches(ds,list(range(len(ds))),3,workers=0))
    b=list(row_batches(ds,list(range(len(ds))),3,workers=2));assert a==b;ds.close()
    x=arguments(tmp_path,'serial');x.epochs=1;train(x)
    y=arguments(tmp_path,'parallel');y.epochs=1;y.workers=2;y.stop_after=2;train(y)
    y.resume=str(tmp_path/'parallel'/'last.pt');y.stop_after=0;y.workers=0;train(y)
    first=read_model(tmp_path/'serial'/'final').state_dict()
    second=read_model(tmp_path/'parallel'/'final').state_dict()
    assert all(torch.equal(first[k],second[k]) for k in first)


def instruction_row():
    return {'id':'train-1','group_id':'original-1','partition':'train','language':'ko',
        'state':'코드는 수정했지만 테스트하지 않았다.','type':'choice','question':'다음 할 일은?',
        'criteria':{'test':'테스트한다.','finish':'완료 처리한다.'},'target_probs':[.9,.1],
        'instruction_views':[{'view_id':'reviewed-a','reviewed':True,'question':'다음 작업으로 무엇을 할까?',
            'criteria':{'finish':'작업을 끝낸다.','test':'수정 내용을 테스트한다.'},'language':'ko'}]}


def test_same_language_instruction_pair_keeps_gold_and_order():
    from nanodecision2.instruction_views import reviewed_pair
    from nanodecision2.multilingual import reviewed_pair_units,bilingual_js
    a,b=reviewed_pair(instruction_row())
    assert a['group_id']==b['group_id'] and a['equivalence_id']==b['equivalence_id']
    assert b['target_probs']==[.1,.9] and a['language']==b['language']=='ko'
    encoded=[]
    for r in (a,b):
        encoded.append({**r,'option_ids':list(r['criteria']),'values':None})
    assert reviewed_pair_units(encoded)==[(0,1)]
    loss,n=bilingual_js(torch.tensor([[2.,-1.],[-1.,2.]]),encoded,torch.tensor([[0,1],[0,1]]))
    assert n==1 and abs(float(loss))<1e-6


@pytest.mark.parametrize('mutation',['unreviewed','new_candidate','test','label_injection','same_prompt'])
def test_instruction_views_reject_unsafe_transforms(mutation):
    from nanodecision2.instruction_views import reviewed_pair
    r=instruction_row();v=r['instruction_views'][0]
    if mutation=='unreviewed':v['reviewed']=False
    elif mutation=='new_candidate':v['criteria']['other']='unknown'
    elif mutation=='test':r['partition']='test'
    elif mutation=='label_injection':v['target']='finish'
    else:v.update(question=r['question'],criteria=r['criteria'])
    with pytest.raises(ValueError):reviewed_pair(r)


def test_instruction_generation_counts_views_not_sources_and_keeps_test(tmp_path):
    from nanodecision2.instruction_views import build_views
    rows=[instruction_row(),{**instruction_row(),'id':'test-2','group_id':'test-2','partition':'test'}]
    path=tmp_path/'in.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=build_views(path,tmp_path/'out.jsonl',fraction=1.)
    actual=[json.loads(x) for x in (tmp_path/'out.jsonl').read_text().splitlines()]
    assert report['source_rows']==2 and report['output_rows']==3 and report['reviewed_pairs']==1
    assert actual[-1]['partition']=='test' and 'equivalence_id' not in actual[-1]


def test_macro_task_accuracy_selection_is_supported(tmp_path):
    from test_training import make_rows,arguments
    from nanodecision2.data import prepare_cache
    from nanodecision2.training import train
    initialize(tmp_path/'init',smoke=True)
    rows=make_rows()
    for i,r in enumerate(rows):r['task']='first' if i%3 else 'second'
    path=tmp_path/'rows.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    prepare_cache([path],tmp_path/'cache',tmp_path/'init',128,128)
    a=arguments(tmp_path,'run');a.epochs=1;a.selection_metric='macro_task_error'
    result=train(a)
    metric=result['history'][-1]['validation']
    assert metric['macro_task_accuracy']==pytest.approx(1-metric['macro_task_error'])
    assert set(metric['by_task'])=={'first','second'}
    assert (tmp_path/'run'/'best'/'model.safetensors').exists()
