import copy
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest
from nanodecision2.auto_sources import select_files,convert,download_url,raw_rows
from nanodecision2.autodata import Store,quotas,build
from nanodecision2.autopipeline import smoke_recipe,smoke_rows,Pipeline,exclusive_workdir
from nanodecision2.initialization import initialize
from nanodecision2.data import CacheDataset
from nanodecision2.train_views import english_gate,training_view,validation_view
from nanodecision2.auto_eval import load_bundle_weights,evaluate
from nanodecision2.export import export_bundle
from nanodecision2.training import read_model

@pytest.fixture
def init(tmp_path):
    p=tmp_path/'init';initialize(p,smoke=True);return p

def test_exact_integer_quotas():
    assert quotas(5000000,{'en':.5,'ko':.4,'other':.1})=={'en':2500000,'ko':2000000,'other':500000}
    assert sum(quotas(7,{'en':.5,'ko':.4,'other':.1}).values())==7

@pytest.mark.parametrize('bad',[{'en':1,'ko':1,'other':0},{'en':float('nan'),'ko':0,'other':1},{'en':1}])
def test_bad_quotas(bad):
    with pytest.raises(ValueError):quotas(500,bad)

def test_parquet_only_requested_split_config():
    files=['ko/train/0000.parquet','ko/test/0000.parquet','en/train/0000.parquet','README.md','data/train-0000.parquet']
    assert select_files(files,'ko','train')==['ko/train/0000.parquet']
    assert select_files(files,None,'train')==['data/train-0000.parquet']
    assert select_files(files,'zh','train')==[]

def test_explicit_hardnegative_variant_not_both():
    assert select_files(['train.jsonl','train_minedHN.jsonl'],None,'train',['train_minedHN.jsonl'])==['train_minedHN.jsonl']

def test_neutral_is_not_hypothesis_false():
    spec=dict(id='kornli-entails',adapter='nli_entails',split='train',language='ko',policy='train_only')
    out=convert(spec,{'premise':'a','hypothesis':'b','label':'neutral'},'1')[0]
    assert out['target'] is False and out['type']=='noul'
    assert '반드시 참' in out['question'] and '가설이 거짓' not in out['criteria']['false']
    assert out['derived'] and out['auto_policy']=='train_only'

def test_mined_pairs_really_different_and_correct():
    spec=dict(id='wiki',adapter='wiki_pairs',split='train',language='ko',max_pairs=12)
    rows=convert(spec,dict(query='query',pos=['positive'],neg=['negative1','negative1','positive','negative2']),'a')
    assert len(rows)==2
    assert len({r['origin_id'] for r in rows})==1
    for r in rows:
        assert r['state']['passages'][r['target']]=='positive'
        assert r['derived'] and r['weight']<1

def test_msmarco_no_unselected_absolute_irrelevance():
    spec=dict(id='ms',adapter='msmarco',split='train',language='en')
    rows=convert(spec,dict(query='q',passages=dict(passage_text=['p','n1','n2'],is_selected=[1,0,0])),'q')
    assert len(rows)==2 and {r['target'] for r in rows}=={True,False}
    assert all('selected' in r['question'] and 'not_exhaustive' in r['label_origin'] for r in rows)

def test_toolace_never_feeds_gold_to_current_request():
    spec=dict(id='ta',adapter='toolace',split='train',language='en')
    raw={'tools':[dict(name='f',description='foo'),dict(name='g',description='bar')],
         'conversations':[dict(**{'from':'human','value':'please foo'}),dict(**{'from':'assistant','value':'[{"name":"f","arguments":{}}]'})]}
    rows=convert(spec,raw,'0')
    assert len(rows)==1 and rows[0]['target']=='f'
    assert len(rows[0]['state']['history'])==1
    raw['conversations'][1]['value']='[{"name":"f"},{"name":"g"}]'
    assert convert(spec,raw,'0')==[]

def test_yelp_ordinal_rating():
    spec=dict(id='yelp',adapter='yelp',split='train',language='en')
    r=convert(spec,dict(text='review',label=4),'a')[0]
    assert r['type']=='score' and r['target']==5


def one_row(i=0,split='train',g=None):
    return dict(id=str(i),group_id=g or 'group'+str(i),origin_id='origin'+str(i),source_split=split,
                state='state '+str(i),type='choice',question='Which?',criteria={'a':'apple','b':'banana'},target='a',language='en',auto_policy='split_train')

def test_group_union_transitive_and_official_test_exclusion(init,tmp_path):
    recipe=smoke_recipe();s=Store(tmp_path/'data',recipe,init);spec=dict(id='fixture',split='train')
    a=one_row(1,g='a');b=one_row(1,split='test',g='b')
    assert s.add(a,spec)=='accepted';assert s.add(b,spec)=='duplicate'
    # A repeated input on test has same label; train copy must NOT enter train.
    s.db.commit();s.assign()
    assert s.db.execute("SELECT count(*) FROM eligible WHERE part='train'").fetchone()[0]==0
    s.close()

def test_conflicting_labels_excluded_not_first_wins(init,tmp_path):
    s=Store(tmp_path/'data',smoke_recipe(),init);spec=dict(id='x',split='train')
    r=one_row();assert s.add(r,spec)=='accepted';r['target']='b'
    assert s.add(r,spec)=='conflicting_label';s.assign()
    assert s.db.execute('SELECT count(*) FROM eligible').fetchone()[0]==0;s.close()

def test_overlength_drops_row_without_truncation(init,tmp_path):
    s=Store(tmp_path/'data',smoke_recipe(),init);r=one_row();r['state']='a'*2000
    assert s.add(r,dict(id='x',split='train'))=='overlength';s.close()

def test_store_resume_rejects_recipe_change(init,tmp_path):
    r=smoke_recipe();s=Store(tmp_path/'data',r,init);s.close()
    r['train_rows']+=1
    with pytest.raises(ValueError):Store(tmp_path/'data',r,init)

def test_shortage_is_not_filled_by_repeats(init,tmp_path):
    s=Store(tmp_path/'data',smoke_recipe(),init)
    for _ in range(30):s.add(one_row(),dict(id='x',split='train'))
    with pytest.raises(RuntimeError,match='Not enough'):s.finalize(tmp_path/'cache')
    assert not (tmp_path/'cache'/'manifest.json').exists();assert (tmp_path/'data'/'shortfall.json').exists();s.close()

def test_full_data_build_exact_counts_and_resume(init,tmp_path):
    r=smoke_recipe();manifest=build(r,tmp_path,init,smoke_rows(r))
    assert manifest['counts']['train']==120
    assert sum(v for k,v in manifest['by_language'].items() if k.startswith('train:'))==120
    assert manifest['by_language']['train:en']==60 and manifest['by_language']['train:ko']==48
    assert manifest['cache_encoding']=='zlib_json' and manifest['smoke_only']
    ds=CacheDataset(tmp_path/'cache','train');ds.verify();assert len(ds)==120
    en=training_view(ds,'en');assert len(en)==60 and all(en[i]['language']=='en' for i in range(60));en.close()
    second=build(r,tmp_path,init,smoke_rows(r));assert second['fingerprint']==manifest['fingerprint']
    groups={}
    for split in manifest['counts']:
        ds=CacheDataset(tmp_path/'cache',split)
        for i in range(len(ds)):
            row=ds[i];assert groups.setdefault(row['group_id'],split)==split
        ds.close()

def test_native_weight_reconstruction(init,tmp_path):
    import torch
    from nanodecision2 import _native
    model=read_model(init);out=tmp_path/'bundle'
    export_bundle(model,out,init,quantize=True,smoke_only=True)
    loaded,manifest=load_bundle_weights(out)
    handle=_native.load(str(out/'model.nd2'))
    from nanodecision2.data import collate
    from nanodecision2.training import INPUTS
    r=dict(state_ids=[2,5,6,1],pair_ids=[[2,10,1],[2,11,1]],target_probs=[1.,0.],type='choice',values=None,weight=1.,option_ids=['a','b'])
    batch=collate([r],0)
    with torch.inference_mode():z=loaded(**{k:batch[k] for k in INPUTS})[0].tolist()
    actual=_native.score(handle,_native.prepare(handle,r['state_ids']),r['pair_ids'])
    assert actual==pytest.approx(z,abs=1e-5)

@pytest.mark.parametrize('accuracy,nll,expected',[(.9,.2,True),(.89,.1,False),(.91,.21,False),(.91,.19,True)])
def test_english_gate(accuracy,nll,expected):
    ref={'by_language':{'en':{'accuracy':.9,'nll':.2}}}
    candidate={'by_language':{'en':{'accuracy':accuracy,'nll':nll}}}
    assert english_gate(candidate,ref)==expected

def test_workdir_lock(tmp_path):
    with exclusive_workdir(tmp_path):
        with pytest.raises(RuntimeError):
            with exclusive_workdir(tmp_path):pass


def test_range_download_resume_and_hash(tmp_path):
    data=b'abcXYZ'*1000;requested=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value=self.headers.get('Range');requested.append(value)
            start=int(value[6:-1]) if value else 0
            self.send_response(206 if value else 200)
            self.send_header('ETag','"fixture"')
            if value:self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
            self.send_header('Content-Length',str(len(data)-start));self.end_headers();self.wfile.write(data[start:])
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    path=tmp_path/'x.gz';partial=tmp_path/'x.gz.partial';partial.write_bytes(data[:99])
    (tmp_path/'x.gz.partial.json').write_text(json.dumps({'etag':'"fixture"'}))
    try:
        url=f'http://127.0.0.1:{server.server_port}/data'
        assert download_url(url,path).read_bytes()==data
        assert requested==['bytes=99-']
        download_url(url,path);assert len(requested)==1
    finally:server.shutdown();server.server_close()

def test_missing_locale_not_extracted(tmp_path):
    p=tmp_path/'bad.tar.gz'
    with tarfile.open(p,'w:gz') as tar:
        content=b'{}\n';info=tarfile.TarInfo('../../unexpected');info.size=len(content);tar.addfile(info,io.BytesIO(content))
    with pytest.raises(ValueError):list(raw_rows(p,{'transport':'massive','language':'ko-KR','split':'train'}))
    assert not (tmp_path.parent/'unexpected').exists()

def test_pipeline_completed_stage_checks_artifacts(tmp_path):
    from nanodecision2.auto_sources import atomic_json,sha_file
    work=tmp_path/'work';work.mkdir()
    p=Pipeline(work,smoke_recipe(),'cpu')
    result=work/'stage_results'/'initialize.json';atomic_json(result,{'ok':True})
    artifact=work/'sentinel.bin';artifact.write_bytes(b'original')
    p.state['stages']['initialize']={'status':'complete','result_sha256':sha_file(result),'artifacts':{str(artifact):sha_file(artifact)}}
    p.stage('initialize')
    artifact.write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='artifact changed'):p.stage('initialize')

def test_pipeline_changed_recipe_refuses_resume(tmp_path):
    from nanodecision2.auto_sources import atomic_json
    work=tmp_path/'work';work.mkdir();r=smoke_recipe();p=Pipeline(work,r,'cpu')
    atomic_json(p.state_path,p.state)
    r['seed']+=1
    with pytest.raises(RuntimeError,match='changed'):Pipeline(work,r,'cpu')

def test_direct_kornli_tsv_adapter(tmp_path):
    f=tmp_path/'official.tsv'
    f.write_text('sentence1\tsentence2\tgold_label\n비가 온다.\t땅이 젖었다.\tneutral\n',encoding='utf-8')
    spec={'id':'kornli-snli','adapter':'nli','split':'train','language':'ko','transport':'url','policy':'train_only'}
    rows=list(raw_rows(f,spec));assert len(rows)==1
    d=convert(spec,rows[0][1],0)
    assert d[0]['target']=='neutral' and d[0]['auto_policy']=='train_only'

def test_no_auto_stage_on_runtime_import():
    # Importing runtime remains light; CLI dispatch is lazy.
    import ast
    text=(Path(__file__).parents[1]/'nanodecision2'/'__init__.py').read_text()
    imports=[n.module for n in ast.walk(ast.parse(text)) if isinstance(n,ast.ImportFrom)]
    assert imports==['runtime']

def test_korean_guard_cannot_be_hidden_by_other_language_gain():
    from nanodecision2.train_views import bilingual_gate
    r={'by_language':{'en':{'accuracy':.8,'nll':.6},'ko':{'accuracy':.7,'nll':.8}}}
    good=copy.deepcopy(r);good['by_language']['ko']={'accuracy':.8,'nll':.6}
    bad=copy.deepcopy(r);bad['by_language']['ko']={'accuracy':.69,'nll':.81}
    assert bilingual_gate(good,r)
    assert not bilingual_gate(bad,r)
