"""Resumable, disk-backed exact-quota data construction. No repeat-to-fill.

Tokenization happens before accepting a row. Source shards, accepted rows, skip
counters and cursors commit together. Final caches use per-record zlib rather
than huge JSON token arrays on disk. All same-state/source-group connections are
reconciled before partitioning, including transitive overlap with official test.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import time
import unicodedata
import zlib
from .schema import TextCodec, normalize_question, gold_target, state_text
from .multilingual import language_key
from .data import SPLITS, digest
from .auto_sources import PublicReader, raw_rows, convert, atomic_json, sha_file


def packed(obj):
    return zlib.compress(json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode(), 1)


def unpacked(blob): return json.loads(zlib.decompress(blob))


def bucket(lang):
    k = language_key(lang)
    return k if k in ('en','ko') else 'other'


def quotas(n, fractions):
    if type(n) is not int or n < 0: raise ValueError('Invalid target count')
    if set(fractions) != {'en','ko','other'} or not all(math.isfinite(x) and x >= 0 for x in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError('Language fractions must be en/ko/other, nonnegative, summing to one')
    out = {k: int(n*v) for k,v in fractions.items()}
    for k in sorted(out, key=lambda k: (-(n*fractions[k]-out[k]), k))[:n-sum(out.values())]: out[k] += 1
    return out


class Store:
    def __init__(self, root: Path, recipe: dict, init_dir: Path):
        self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.recipe=recipe; self.init=Path(init_dir)
        self.info=json.loads((self.init/'init.json').read_text())
        self.cfg=json.loads((self.init/'config.json').read_text())
        self.tokenizer_hash = 'smoke_byte_v1' if self.info.get('smoke_only') else sha_file(self.init/'tokenizer.json')
        self.codec=TextCodec(self.init/'tokenizer.json', smoke=self.info.get('smoke_only',False))
        self.db=sqlite3.connect(self.root/'pool.sqlite', timeout=120)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('PRAGMA cache_size=-65536'); self.db.execute('PRAGMA temp_store=FILE')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata(k TEXT PRIMARY KEY, v TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS groups(g TEXT PRIMARY KEY, parent TEXT NOT NULL, flags INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS states(s TEXT PRIMARY KEY, g TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS rows(k TEXT PRIMARY KEY, label TEXT NOT NULL, g TEXT NOT NULL,
              origin TEXT NOT NULL, source TEXT NOT NULL, original_split TEXT NOT NULL,
              language TEXT NOT NULL, bucket TEXT NOT NULL, priority INTEGER NOT NULL,
              raw BLOB NOT NULL, encoded BLOB NOT NULL, length INTEGER NOT NULL,
              derived INTEGER NOT NULL, task TEXT NOT NULL, blocked INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS progress(source TEXT PRIMARY KEY, shard INTEGER NOT NULL,
              next_row INTEGER NOT NULL, accepted INTEGER NOT NULL, done INTEGER NOT NULL,
              raw_seen INTEGER NOT NULL, stats TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS shards(source TEXT, shard INTEGER, info TEXT, PRIMARY KEY(source,shard));
        ''')
        signature=digest(json.dumps({'recipe':recipe,'tokenizer':self.tokenizer_hash,'config':self.cfg},sort_keys=True))
        old=self.db.execute("SELECT v FROM metadata WHERE k='signature'").fetchone()
        if old and old[0]!=signature: raise ValueError('Recipe/tokenizer changed. Resume refused; use a new work directory.')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature',?)",(signature,));self.db.commit()
        self.signature=signature

    def close(self): self.db.commit(); self.db.close()

    def find(self, g):
        visited=[]
        while True:
            parent=self.db.execute('SELECT parent FROM groups WHERE g=?',(g,)).fetchone()
            if parent is None: raise RuntimeError('Missing source group')
            if parent[0]==g: break
            visited.append(g);g=parent[0]
        for child in visited: self.db.execute('UPDATE groups SET parent=? WHERE g=?',(g,child))
        return g

    def union(self, a, b):
        a,b=self.find(a),self.find(b)
        if a==b:return a
        root,child=sorted((a,b))
        flags=self.db.execute('SELECT flags FROM groups WHERE g=?',(child,)).fetchone()[0]
        self.db.execute('UPDATE groups SET parent=? WHERE g=?',(root,child))
        self.db.execute('UPDATE groups SET flags=flags|? WHERE g=?',(flags,root))
        return root

    def add(self, row, spec):
        text=state_text(row['state'])
        if not text.strip() or len(text)>262144: return 'empty_or_excessive_state'
        try:
            q=normalize_question(row);target=gold_target(row,q)
            if len(q['options'])>self.recipe.get('max_candidates',20): return 'too_many_candidates'
            sids=self.codec.state(row['state'],self.recipe['max_state_tokens'])
            pairs=self.codec.pairs(q,self.recipe['max_pair_tokens'])
        except ValueError as e:
            if 'No silent truncation' in str(e): return 'overlength'
            raise  # A malformed mapping is a code/schema problem, not a silent skip.
        w=float(row.get('weight',1.0))
        if not math.isfinite(w) or w<=0:raise ValueError('Invalid data weight')
        g=digest(row['group_id']);part=row.get('source_split',spec['split'])
        flags=(4 if part=='test' else 2 if part in ('validation','dev','validation_matched','validation_mismatched') else 1)
        if flags==1 and row.get('auto_policy',spec.get('policy'))=='train_only':flags|=8
        self.db.execute('INSERT OR IGNORE INTO groups VALUES (?,?,?)',(g,g,flags))
        root=self.find(g);self.db.execute('UPDATE groups SET flags=flags|? WHERE g=?',(flags,root))
        state_hash=digest(unicodedata.normalize('NFKC',text).strip())
        old=self.db.execute('SELECT g FROM states WHERE s=?',(state_hash,)).fetchone()
        if old:self.union(g,old[0])
        else:self.db.execute('INSERT INTO states VALUES (?,?)',(state_hash,g))
        ordered=sorted((o['description'].strip(),None if q['values'] is None else q['values'][i],target[i]) for i,o in enumerate(q['options']))
        identity={'state':unicodedata.normalize('NFKC',text).strip(),'question':q['question'].strip(), 'type':q['type'],
                  'options':[(a,b) for a,b,c in ordered]}
        key=digest(json.dumps(identity,ensure_ascii=False,sort_keys=True))
        label=json.dumps([c for a,b,c in ordered])
        prev=self.db.execute('SELECT label,g FROM rows WHERE k=?',(key,)).fetchone()
        if prev:
            self.union(g,prev[1])
            if prev[0]!=label:
                self.db.execute('UPDATE rows SET blocked=1 WHERE k=?',(key,));return 'conflicting_label'
            return 'duplicate'
        lang=language_key(row.get('language')); origin=row.get('origin_id',spec['id']+':'+row['id'])
        encoded={'id':key,'group_id':g,'type':q['type'],'state_ids':sids,'pair_ids':pairs,'target_probs':target,
                 'option_ids':[o['id'] for o in q['options']], 'values':q['values'],'weight':w,
                 'task':row.get('task',q['type']),'language':lang,'source':row.get('source',spec['id']),
                 'label_origin':row.get('label_origin','unspecified'),'reference_option_id':row.get('reference_option_id'),
                 'equivalence_id':None,'equivalence_verified':False,'origin_id':origin,'derived':row.get('derived',False)}
        n_tokens=len(sids)+sum(map(len,pairs))
        self.db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)',
            (key,label,g,origin,spec['id'],part,lang,bucket(lang),spec.get('priority',10),packed(row),packed(encoded),
             n_tokens,int(row.get('derived',False)),row.get('task',q['type'])))
        return 'accepted'

    def progress(self, source):
        row=self.db.execute('SELECT shard,next_row,accepted,done,raw_seen,stats FROM progress WHERE source=?',(source,)).fetchone()
        return dict(zip(('shard','next_row','accepted','done','raw_seen','stats'),row)) if row else dict(shard=0,next_row=0,accepted=0,done=0,raw_seen=0,stats='{}')

    def save_progress(self,source,p,stats):
        self.db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?,?,?,?,?)',
            (source,p['shard'],p['next_row'],p['accepted'],p['done'],p['raw_seen'],json.dumps(dict(stats))))
        self.db.commit()

    def ingest(self, spec, reader: PublicReader):
        p=self.progress(spec['id']);stats=Counter(json.loads(p['stats']))
        if p['done']:return stats
        cap=spec.get('max_decisions',0)
        if cap and p['accepted']>=cap:
            p['done']=1;self.save_progress(spec['id'],p,stats);return stats
        started=time.monotonic()
        for si,path,provenance in reader.shards(spec):
            if si<p['shard']:continue
            shard_sig=json.dumps(provenance,sort_keys=True)
            old=self.db.execute('SELECT info FROM shards WHERE source=? AND shard=?',(spec['id'],si)).fetchone()
            if old and old[0]!=shard_sig:raise ValueError('Downloaded shard changed since checkpoint')
            self.db.execute('INSERT OR IGNORE INTO shards VALUES (?,?,?)',(spec['id'],si,shard_sig))
            begin=p['next_row'] if si==p['shard'] else 0
            reading=dict(spec)
            if provenance.get('file','').endswith('.parquet'):reading['_parquet']=True
            for i,raw,names in raw_rows(path,reading,begin):
                converted=convert(spec,raw,f'{si}:{i}',names)
                for row in converted:
                    row['source_revision']=provenance
                    result=self.add(row,spec);stats[result]+=1
                    if result=='accepted':p['accepted']+=1
                p.update(shard=si,next_row=i+1,raw_seen=p['raw_seen']+1)
                if p['raw_seen']%1000==0:
                    self.save_progress(spec['id'],p,stats)
                    print(json.dumps({'stage':'data','source':spec['id'],'raw_seen':p['raw_seen'],
                                      'accepted':p['accepted'],'rejections':dict(stats),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
                if cap and p['accepted']>=cap:break
            self.save_progress(spec['id'],p,stats)
            if cap and p['accepted']>=cap:break
            p.update(shard=si+1,next_row=0);self.save_progress(spec['id'],p,stats)
        p['done']=1;self.save_progress(spec['id'],p,stats)
        return stats

    def assign(self):
        # Rebuilt from committed pool on each invocation; never trust stale selections.
        self.db.executescript('DROP TABLE IF EXISTS roots; CREATE TABLE roots(g TEXT PRIMARY KEY, root TEXT, part TEXT, flags INTEGER);')
        cursor=self.db.execute('SELECT g FROM groups ORDER BY g')
        n=0
        while True:
            batch=cursor.fetchmany(2000)
            if not batch:break
            for (g,) in batch:
                root=self.find(g);flags=self.db.execute('SELECT flags FROM groups WHERE g=?',(root,)).fetchone()[0]
                v=int(digest(str(self.recipe['seed'])+':'+root)[:16],16)/2**64
                if flags&4:part='test'
                elif flags&2:part='validation'
                elif flags&8:part='train'
                else:part='train' if v<.94 else 'validation' if v<.96 else 'temperature' if v<.98 else 'conformal'
                self.db.execute('INSERT INTO roots VALUES (?,?,?,?)',(g,root,part,flags));n+=1
            self.db.commit()
            if n%100000==0:print(json.dumps({'stage':'reconcile_groups','groups_processed':n}),flush=True)
        self.db.executescript('''
          DROP TABLE IF EXISTS eligible;
          CREATE TABLE eligible AS SELECT r.k,r.bucket,r.priority,r.g,t.root,t.part,r.language,r.source,r.task,r.origin,r.derived,r.length
          FROM rows r JOIN roots t ON r.g=t.g WHERE r.blocked=0 AND
          ((t.flags&4!=0 AND r.original_split='test') OR
           (t.flags&4=0 AND t.flags&2!=0 AND r.original_split IN ('validation','dev','validation_matched','validation_mismatched')) OR
           (t.flags&6=0 AND r.original_split='train'));
          CREATE INDEX eligible_selection ON eligible(part,bucket,priority,k);
        ''')
        self.db.commit()

    def finalize(self, out: Path):
        out=Path(out)
        if (out/'manifest.json').exists():
            manifest=json.loads((out/'manifest.json').read_text())
            if manifest.get('build_signature')!=self.signature:raise ValueError('Cache belongs to another build')
            return manifest
        self.assign()
        desired={'train':self.recipe['train_rows'],**self.recipe['reserve_rows']}
        target={s:quotas(desired[s],self.recipe['language_fractions']) for s in SPLITS}
        available={(p,b):n for p,b,n in self.db.execute('SELECT part,bucket,count(*) FROM eligible GROUP BY part,bucket')}
        shortages={f'{p}:{b}':{'needed':n,'available':available.get((p,b),0)} for p,qs in target.items() for b,n in qs.items() if available.get((p,b),0)<n}
        if shortages:
            atomic_json(self.root/'shortfall.json',{'shortfalls':shortages,'targets':target,
                'action':'Training is blocked. No repeated rows, changed language ratios, or reduced target counts were used.'})
            raise RuntimeError('Not enough eligible unique decisions. See data/shortfall.json. '+json.dumps(shortages))
        self.db.executescript('DROP TABLE IF EXISTS selected; CREATE TABLE selected(k TEXT PRIMARY KEY, part TEXT, root TEXT);')
        for part, qs in target.items():
            for b, n in qs.items():
                self.db.execute('INSERT INTO selected SELECT k,part,root FROM eligible WHERE part=? AND bucket=? ORDER BY priority,k LIMIT ?', (part,b,n))
        self.db.commit()
        # Verify using original group components, not merely row IDs.
        leaks=self.db.execute('SELECT root FROM selected GROUP BY root HAVING count(DISTINCT part)>1 LIMIT 1').fetchone()
        if leaks:raise RuntimeError('Internal group partition leakage')
        temporary=out.with_name(out.name+'.building')
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)  # Dedicated generated staging directory only.
        temporary.mkdir(parents=True)
        counts=Counter();tasks=Counter();langs=Counter();sources=Counter();origins=Counter();derived=0;total_tokens=0
        group_hash=hashlib.sha256()
        for root,part in self.db.execute('SELECT DISTINCT root,part FROM selected ORDER BY root'):
            group_hash.update((root+':'+part+'\n').encode())
        for part in SPLITS:
            with (temporary/(part+'.records')).open('wb') as data,(temporary/(part+'.idx')).open('wb') as idx,(temporary/(part+'.lengths')).open('wb') as lengths,(temporary/(part+'.jsonl.gz')).open('wb') as rawfile:
                with gzip_writer(rawfile) as rawgz:
                    query='''SELECT r.raw,r.encoded,r.length,s.root,r.language,r.source,r.task,r.derived
                             FROM selected s JOIN rows r ON r.k=s.k WHERE s.part=? ORDER BY s.k'''
                    for rawblob,encblob,length,root,lang,src,task,isderived in self.db.execute(query,(part,)):
                        enc=unpacked(encblob);enc['group_id']=root;enc['split']=part
                        blob=packed(enc)
                        idx.write(struct.pack('<Q',data.tell()));data.write(blob);lengths.write(struct.pack('<I',length))
                        raw=unpacked(rawblob);raw.update(partition=part,group_id=root,id=enc['id'])
                        rawgz.write((json.dumps(raw,ensure_ascii=False,separators=(',',':'))+'\n').encode())
                        counts[part]+=1;total_tokens+=length
                        langs[part+':'+lang]+=1;sources[part+':'+src]+=1;tasks[part+':'+task]+=1
                        if part=='train':derived+=isderived
                        if counts[part]%100000==0:print(json.dumps({'stage':'write_cache','split':part,'rows':counts[part]}),flush=True)
                for f in (data,idx,lengths,rawfile):f.flush();os.fsync(f.fileno())
        files={s:{suffix:sha_file(temporary/(s+suffix)) for suffix in ('.records','.idx','.lengths')} for s in SPLITS}
        independent_origins=self.db.execute("SELECT count(DISTINCT r.origin) FROM selected s JOIN rows r ON s.k=r.k WHERE s.part='train'").fetchone()[0]
        source_groups=self.db.execute("SELECT count(DISTINCT root) FROM selected WHERE part='train'").fetchone()[0]
        status={src:{**self.progress(src),'stats':json.loads(self.progress(src)['stats'])} for (src,) in self.db.execute('SELECT source FROM progress')}
        manifest={'version':2,'cache_encoding':'zlib_json','data_suffix':'.records','build_signature':self.signature,
            'counts':dict(counts),'requested_train_rows':self.recipe['train_rows'],'language_targets':target,
            'by_language':dict(langs),'by_source':dict(sources),'by_task':dict(tasks),
            'train_original_records':independent_origins,'train_source_groups':source_groups,'train_derived_decisions':derived,
            'group_assignment_sha256':group_hash.hexdigest(),'input_files_sha256':sorted(sha_file(temporary/(s+'.jsonl.gz')) for s in SPLITS),
            'encoded_tokens':total_tokens,'max_state_tokens':self.recipe['max_state_tokens'],'max_pair_tokens':self.recipe['max_pair_tokens'],
            'tokenizer_sha256':self.tokenizer_hash,'files':files,'seed':self.recipe['seed'],
            'smoke_only':self.info.get('smoke_only',False),'source_statistics':status,
            'source_shards':[{'source':src,'shard':shard,**json.loads(info)} for src,shard,info in self.db.execute('SELECT source,shard,info FROM shards ORDER BY source,shard')],
            'split_policy':'Official-test/development overlap excludes training groups; original NLI train and all its translations remain train-only; all other source groups hash 94/2/2/2.',
            'limitations':'5M DECISIONS != 5M originals. Exact-state/source-group dedup only, not complete semantic or translation dedup. Mined negatives and published synthetic labels are not human truth.'}
        manifest['fingerprint']=digest(json.dumps(manifest,sort_keys=True))
        manifest['disk_bytes']=sum(p.stat().st_size for p in temporary.iterdir())
        atomic_json(temporary/'manifest.json',manifest)
        if out.exists():raise FileExistsError('Final cache path already exists without manifest')
        os.replace(temporary,out);return manifest


def gzip_writer(file):
    import gzip
    return gzip.GzipFile(fileobj=file,mode='wb',compresslevel=1,mtime=0)


def build(recipe,work,init_dir,local_sources=None):
    work=Path(work);store=Store(work/'data',recipe,Path(init_dir));reader=PublicReader(work/'downloads')
    errors=[]
    try:
        for spec in recipe['sources']:
            if spec['adapter']=='nli_entails' and not recipe.get('derived_nli_entailment',False):continue
            try:
                if local_sources is None:
                    store.ingest(spec,reader)
                    total=store.db.execute('SELECT coalesce(sum(accepted),0) FROM progress').fetchone()[0]
                    if total>recipe.get('max_candidate_rows',10000000):raise RuntimeError('Candidate pool safety limit exceeded; target counts were not changed')
                else:
                    p=store.progress(spec['id']);stats=Counter(json.loads(p['stats']))
                    if p['done']:continue
                    for i,row in enumerate(local_sources[spec['id']]):
                        if i<p['next_row']:continue
                        stats[store.add(row,spec)]+=1;p['accepted']=stats['accepted'];p['raw_seen']+=1;p['next_row']=i+1
                    p['done']=1;store.save_progress(spec['id'],p,stats)
            except Exception as exc:
                store.db.rollback()
                errors.append({'source':spec['id'],'error_type':type(exc).__name__,'error':str(exc),'required':spec.get('required',True)})
                atomic_json(work/'source_errors.json',errors)
                if spec.get('required',True):raise
                print(json.dumps({'stage':'optional_source_failed',**errors[-1]}),flush=True)
        manifest=store.finalize(work/'cache')
        atomic_json(work/'data_report.json',manifest)
        return manifest
    finally:store.close()
