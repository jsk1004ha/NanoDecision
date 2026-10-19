"""Pinned public-data readers and faithful decision adapters for the 0.3.1 autopilot.

No API inference, no remote Python dataset scripts. Hub files are downloaded by
shard with huggingface_hub's cache/resume, then read using pyarrow/json/csv.
The converted Parquet branch (when necessary) gets its OWN immutable revision.
"""
from __future__ import annotations
import csv
import fnmatch
import gzip
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import random
import tarfile
import time
import urllib.error
import urllib.request
from typing import Any, Iterator
from . import adapters as A
from .schema import state_text

MASSIVE_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz'


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def retry(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            # Authentication, missing resources and invalid schemas aren't transient.
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (400, 401, 403, 404) or attempt + 1 == tries:
                raise
            time.sleep(min(30, 2 ** attempt))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def download_url(url: str, path: Path, attempts: int = 5) -> Path:
    """Range resume with ETag/If-Range. A .partial never counts as a finished file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(path.suffix + '.download.json')
    if path.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        if info['url'] == url and info['sha256'] == sha_file(path): return path
        raise ValueError(f'Download changed: {path}; use a new work directory')
    tmp = path.with_name(path.name + '.partial')
    meta = tmp.with_suffix(tmp.suffix + '.json')
    for attempt in range(attempts):
        try:
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {'User-Agent': 'NanoDecision2/0.3.1'}
            saved = json.loads(meta.read_text()) if meta.exists() else {}
            if offset and not saved.get('etag'):
                offset=0
            if offset:
                headers['Range'] = f'bytes={offset}-'
                if saved.get('etag'): headers['If-Range'] = saved['etag']
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                status = getattr(response, 'status', 200)
                if status == 206 and offset:
                    cr = response.headers.get('Content-Range', '')
                    if not cr.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Invalid partial response offset')
                else:
                    offset = 0
                atomic_json(meta, {'etag': response.headers.get('ETag'), 'url': url})
                length = response.headers.get('Content-Length')
                received = 0
                with tmp.open('ab' if offset else 'wb') as f:
                    while True:
                        b = response.read(1 << 20)
                        if not b: break
                        f.write(b); received += len(b)
                    f.flush(); os.fsync(f.fileno())
                if length is not None and received != int(length):
                    raise IOError('Incomplete HTTP body')
            os.replace(tmp, path)
            atomic_json(manifest, {'url': url, 'bytes': path.stat().st_size, 'sha256': sha_file(path)})
            meta.unlink(missing_ok=True)
            return path
        except urllib.error.HTTPError as e:
            if e.code == 416:  # Re-download safely; never infer completion from 416.
                tmp.unlink(missing_ok=True); meta.unlink(missing_ok=True)
            elif e.code in (401, 403, 404): raise
            if attempt + 1 == attempts: raise
            time.sleep(min(30, 2 ** attempt))
        except (OSError, RuntimeError, http.client.IncompleteRead):
            if attempt + 1 == attempts: raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError('Unreachable downloader state')


def select_files(files: list[str], config: str | None, split: str,
                 explicit: list[str] | None = None) -> list[str]:
    """Do not accidentally mix train with test, or concatenate two source variants."""
    candidates = [x for x in files if x.endswith(('.parquet', '.jsonl', '.jsonl.gz', '.json', '.tsv', '.tsv.gz'))]
    if explicit:
        return sorted(x for x in candidates if any(fnmatch.fnmatch(x, p) for p in explicit))
    cfg = config or 'default'
    sets = [
        [f'{cfg}/{split}/*.parquet'],
        [f'{cfg}/{split}-*.parquet', f'{cfg}/{split}.parquet', f'{cfg}/{split}.jsonl', f'{cfg}/{split}.json'],
        [f'{cfg}/data/{split}-*.parquet'],
    ]
    if config in (None, 'default'):
        sets += [[f'data/{split}-*.parquet'], [f'{split}-*.parquet', f'{split}.parquet', f'{split}.jsonl', f'{split}.json'],
                 [f'{split}/*.parquet']]
    for patterns in sets:
        hits = sorted(x for x in candidates if any(fnmatch.fnmatch(x, p) for p in patterns))
        if hits: return hits
    return []


class PublicReader:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / 'sources.lock.json'
        self.lock = json.loads(self.lock_path.read_text()) if self.lock_path.exists() else {}

    def resolve(self, spec: dict) -> dict:
        key = spec['id']
        requested = {k: spec.get(k) for k in ('repo', 'config', 'split', 'revision', 'files', 'transport', 'language', 'url')}
        if key in self.lock:
            if self.lock[key]['requested'] != requested:
                raise ValueError(f'Source configuration changed: {key}. Use a new work directory.')
            return self.lock[key]
        if spec.get('transport') == 'url':
            record = {'requested': requested, 'url': spec['url'], 'files': [spec['url'].rsplit('/',1)[-1]]}
        elif spec.get('transport') == 'massive':
            record = {'requested': requested, 'url': MASSIVE_URL, 'files': ['massive-1.1.tar.gz']}
        else:
            from huggingface_hub import HfApi
            api = HfApi()
            repo = spec['repo']
            revision = retry(lambda: api.dataset_info(repo, revision=spec.get('revision', 'main')).sha)
            names = retry(lambda: api.list_repo_files(repo, repo_type='dataset', revision=revision))
            files = select_files(names, spec.get('config'), spec['split'], spec.get('files'))
            branch = spec.get('revision', 'main')
            if not files:
                # Hub-maintained Parquet only. No trust_remote_code workaround.
                branch = 'refs/convert/parquet'
                revision = retry(lambda: api.dataset_info(repo, revision=branch).sha)
                names = retry(lambda: api.list_repo_files(repo, repo_type='dataset', revision=revision))
                files = select_files(names, spec.get('config'), spec['split'])
            if not files:
                raise RuntimeError(f'No safe data files for {repo}/{spec.get("config")}/{spec["split"]}. '
                                   'The source schema may have changed; no substitute data was used.')
            record = {'requested': requested, 'repo': repo, 'data_revision': revision,
                      'data_branch': branch, 'files': files}
        self.lock[key] = record; atomic_json(self.lock_path, self.lock)
        return record

    def shards(self, spec: dict) -> Iterator[tuple[int, Path, dict]]:
        entry = self.resolve(spec)
        if spec.get('transport') == 'url':
            # The first complete download is frozen by its content hash. Reuse it on resume.
            path = download_url(entry['url'], self.root / 'direct' / entry['files'][0])
            yield 0, path, {'sha256': sha_file(path), 'url': entry['url'], 'file': entry['files'][0]}
        elif spec.get('transport') == 'massive':
            path = download_url(MASSIVE_URL, self.root / 'massive-1.1.tar.gz')
            yield 0, path, {'sha256': sha_file(path), 'url': MASSIVE_URL}
        else:
            from huggingface_hub import hf_hub_download
            for index, filename in enumerate(entry['files']):
                path = Path(retry(lambda: hf_hub_download(entry['repo'], filename, repo_type='dataset',
                                     revision=entry['data_revision'], cache_dir=str(self.root / 'hub'))))
                yield index, path, {'file': filename, 'revision': entry['data_revision'], 'sha256': sha_file(path)}


def raw_rows(path: Path, spec: dict, start: int = 0) -> Iterator[tuple[int, dict, list[str] | None]]:
    """start is the next source record; Parquet resumes without rereading old row groups."""
    if spec.get('transport') == 'massive':
        with tarfile.open(path, 'r:gz') as tar:
            name = 'data/' + spec['language'] + '.jsonl'
            members = [m for m in tar.getmembers() if m.isfile() and
                       (m.name.lstrip('./') == name or m.name.endswith('/' + name))]
            if len(members) != 1: raise ValueError('MASSIVE locale file missing/ambiguous')
            with io.TextIOWrapper(tar.extractfile(members[0]), encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= start and line.strip():
                        r = json.loads(line)
                        if r['partition'] == spec['split']: yield i, r, None
        return
    if str(path).endswith('.parquet') or spec.get('_parquet'):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        names = None
        meta = (pf.schema_arrow.metadata or {}).get(b'huggingface')
        if meta:
            features = json.loads(meta).get('info', {}).get('features', {})
            names = features.get('label', {}).get('names')
        offset = 0
        for group in range(pf.num_row_groups):
            n = pf.metadata.row_group(group).num_rows
            if offset + n <= start:
                offset += n; continue
            for batch in pf.iter_batches(batch_size=512, row_groups=[group]):
                for r in batch.to_pylist():
                    if offset >= start: yield offset, r, names
                    offset += 1
        return
    if str(path).endswith('.json'):
        if path.stat().st_size > (256 << 20):
            raise RuntimeError('Large JSON-array source requires a Parquet release; refusing unbounded RAM use')
        with path.open(encoding='utf-8') as f: obj=json.load(f)
        rows=obj if isinstance(obj,list) else obj.get('data',[])
        if not isinstance(rows,list):raise ValueError('Expected source record list')
        for i,row in enumerate(rows):
            if i>=start:yield i,row,None
        return
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as f:
        rows = csv.DictReader(f, delimiter='\t') if '.tsv' in str(path) else (json.loads(x) for x in f if x.strip())
        for i, row in enumerate(rows):
            if i >= start: yield i, row, None


def _base(spec, index, group, lang=None):
    lang = lang or spec.get('language', 'en')
    r = A.base_row(spec['id'], spec['split'], str(index), group, lang)
    r['source'] = spec.get('repo', spec['id'])
    r['origin_id'] = spec['id'] + ':' + str(index)
    r['auto_policy'] = spec.get('policy', 'split_train')
    r['derived'] = False
    return r


def _pairwise(spec, index, query, positive, negative, pair_no):
    if positive.strip() == negative.strip(): return None
    key = A.fingerprint(query + '\n' + positive + '\n' + negative)
    flip = int(key[:2], 16) % 2
    texts = [positive, negative] if not flip else [negative, positive]
    ko = spec.get('language') == 'ko'
    row = _base(spec, f'{index}:{pair_no}', 'retrieval-query:' + A.fingerprint(query))
    row['origin_id'] = spec['id'] + ':' + str(index)
    row.update(type='choice', task='evidence_pair',
        state={'query': query, 'passages': {'A': texts[0], 'B': texts[1]}},
        question='어느 문서가 질문에 대한 근거로 더 적합한가?' if ko else 'Which passage is the better supporting evidence for the query?',
        criteria={'A': 'A 문서' if ko else 'Passage A', 'B': 'B 문서' if ko else 'Passage B'},
        target='B' if flip else 'A', derived=True, label_origin='source_positive_vs_mined_negative_not_exhaustive_relevance', weight=0.5)
    return row


def convert(spec: dict, raw: dict, index: str, label_names=None) -> list[dict]:
    """Conversion never uses model outputs, random labels, or gold-assisted inputs."""
    kind = spec['adapter']; split = spec['split']; lang = spec.get('language', 'en')
    if kind in ('nli', 'nli_entails'):
        r = dict(raw)
        if 'sentence1' in r: r['premise'], r['hypothesis'] = r['sentence1'], r['sentence2']
        if 'gold_label' in r: r['label'] = r['gold_label']
        if r.get('label') in (-1, '-', None): return []
        names = label_names or spec.get('label_names')
        rows = A.nli_row(r, spec['id'], split, lang, names, original_id=index)
        for row in rows:
            row.update(origin_id=spec.get('origin_family', spec['id'])+':'+A.fingerprint(state_text(row['state'])),
                       auto_policy=spec.get('policy', 'train_only'), derived=kind == 'nli_entails')
            if kind == 'nli_entails':
                # Neutral means 'not entailed', NEVER 'the hypothesis is false'.
                yes = row['target'] == 'entailment'
                row.update(type='noul', task='entailment_check', target=yes,
                    question='전제로부터 가설이 반드시 참이라고 결론내릴 수 있는가?' if lang == 'ko' else 'Does the premise necessarily entail the hypothesis?',
                    criteria={'false':'가설이 반드시 참이라고 결론낼 수 없다.', 'true':'가설이 반드시 참임이 보장된다.'} if lang == 'ko' else
                             {'false':'The premise does not entail the hypothesis.', 'true':'The premise entails the hypothesis.'},
                    label_origin='deterministic_entailment_marginal_of_NLI', weight=0.5)
            elif (lang != 'en' and 'kornli' in spec['id']) or spec.get('translated'):
                row['label_origin'] = 'machine_translated_NLI'
        return rows
    if kind == 'paws':
        rows = A.paws_row(raw, split, lang)
    elif kind == 'massive':
        rows = A.massive_row(raw, 'scenario')
    elif kind in ('ag-news', 'klue-ynat', 'enron-spam', 'phishing', 'support-queue'):
        rows = A.simple_source_rows(raw, kind, split, index)
    elif kind == 'nsmc':
        text = raw.get('document') or raw.get('text')
        if not text or raw.get('label') not in (0, 1, '0', '1'): return []
        row = _base(spec, index, 'review:' + A.fingerprint(text))
        row.update(type='noul', state=text, question='이 영화 리뷰의 감정은 긍정적인가?',
                   criteria={'false':'부정적인 평가다.', 'true':'긍정적인 평가다.'},
                   target=bool(int(raw['label'])), task='sentiment')
        rows = [row]
    elif kind == 'yelp':
        text = raw['text']; score = int(raw['label'])
        if score not in range(5): return []
        row = _base(spec, index, 'yelp-review:' + A.fingerprint(text))
        row.update(type='score', state=text, question="What star rating did this review express?",
                   levels=[{'value':i+1,'description':f'{i+1} star'+('s' if i else '')} for i in range(5)],
                   target=score+1, task='review_rating', label_origin='reviewer_star_rating_not_objective_quality')
        rows = [row]
    elif kind == 'msmarco':
        q = raw['query']; ps = raw['passages']
        texts, labels = ps['passage_text'], ps['is_selected']
        positives = [t for t, y in zip(texts, labels) if y == 1]
        negatives = [t for t, y in zip(texts, labels) if y == 0]
        if not positives or not negatives: return []
        # Balanced selected/not-selected labels; not a claim of complete relevance.
        rows = []
        for j, (text, yes) in enumerate([(t, True) for t in positives] + [(t, False) for t in negatives[:max(1, len(positives))]]):
            row = _base(spec, f'{index}:{j}', 'msmarco-query:' + A.fingerprint(q))
            row.update(origin_id=spec['id']+':'+str(index), type='noul', state={'query':q, 'passage':text},
                question='Was this passage selected as supporting evidence for the query?', target=yes,
                task='passage_selection', derived=True, weight=0.5,
                label_origin='source_passage_selection_not_exhaustive_relevance')
            rows.append(row)
    elif kind == 'wiki_pairs':
        query = raw['query']; positives = raw['pos']; negatives = raw['neg']
        if not positives or not negatives: return []
        # Use different negative passages, not repeated strings or synonym prompts.
        unique = list(dict.fromkeys(t for t in negatives if t not in positives))
        rows = [_pairwise(spec,index,query,positives[0],n,j) for j,n in enumerate(unique[:spec.get('max_pairs', 12)])]
        rows = [r for r in rows if r]
    elif kind == 'toolace':
        rows = toolace_rows(spec, raw, index)
    else:
        raise ValueError('Unknown auto adapter: ' + kind)
    for row in rows:
        row.setdefault('origin_id', spec['id'] + ':' + str(index))
        row.setdefault('derived', False)
        row['auto_policy'] = spec.get('policy', 'split_train')
        row['source'] = spec.get('repo', spec['id'])
    return rows


def toolace_rows(spec, raw, index):
    """Conservative extraction. Parse published tool inventory; never regex-guess JSON."""
    inventory = raw.get('tools')
    if inventory is None:
        system = raw.get('system', '')
        # ToolACE publishes a JSON function inventory inside its system prompt.
        decoder = json.JSONDecoder(); inventory = None
        for pos, ch in enumerate(system):
            if ch not in '[{': continue
            try: obj, _ = decoder.raw_decode(system[pos:])
            except ValueError: continue
            if isinstance(obj, list) and obj and all(isinstance(x, dict) and ('name' in x or 'function' in x) for x in obj):
                inventory = obj; break
        if inventory is None: return []
    if isinstance(inventory, str): inventory = json.loads(inventory)
    tools = [x.get('function', x) for x in inventory]
    if not 2 <= len(tools) <= 20: return []
    if any(not isinstance(t,dict) or not t.get('name') or not t.get('description') for t in tools): return []
    if len({t['name'] for t in tools}) != len(tools): return []
    messages = raw.get('conversations', raw.get('messages', [])); history=[]; result=[]
    for turn, message in enumerate(messages):
        role=message.get('from',message.get('role')); text=message.get('value',message.get('content',''))
        if role in ('assistant','gpt'):
            try:
                calls = message.get('tool_calls') or json.loads(text)
            except (ValueError, TypeError): calls = None
            if isinstance(calls, dict): calls=[calls]
            if isinstance(calls,list) and len(calls)==1 and isinstance(calls[0],dict):
                call=calls[0].get('function', calls[0]); name=call.get('name')
                if name in {t['name'] for t in tools} and any(m['role'] in ('user','human') for m in history):
                    row = _base(spec, f'{index}:{turn}', 'tool-dialogue:'+str(index))
                    row.update(origin_id=spec['id']+':'+str(index), type='choice', task='tool_route',
                        state={'history':list(history)}, question='Which single tool should be called next?',
                        choices=[{'id':t['name'],'description':t['name']+': '+t['description']+'\nParameters: '+json.dumps(t.get('parameters',{}),ensure_ascii=False,sort_keys=True)} for t in tools],
                        target=name, label_origin='published_synthetic_tool_call', derived=True)
                    result.append(row)
        history.append({'role':str(role), 'content':text})
    return result
