"""Bounded, ordered tokenization workers; group/split audits keep one writer."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from .schema import TextCodec, normalize_question, gold_target

_CODEC = None
_LIMITS = None


def _init_codec(source, max_state, max_pair):
    global _CODEC, _LIMITS
    root = Path(source)
    info = json.loads((root/'init.json').read_text())
    _CODEC = TextCodec(root/'tokenizer.json', smoke=info.get('smoke_only',False))
    _LIMITS = max_state, max_pair


def _encode_chunk(chunk):
    result = []
    for path, number, row in chunk:
        try:
            spec = normalize_question(row); gold_target(row,spec)
            result.append((path,number,row,_CODEC.state(row['state'],_LIMITS[0]),
                           _CODEC.pairs(spec,_LIMITS[1])))
        except Exception as exc:
            # Do not include document content or token values in exception text.
            raise ValueError(f'{path}:{number}: tokenization/schema failure ({type(exc).__name__})') from None
    return result


def source_rows(paths):
    for path in paths:
        with open(path,encoding='utf-8') as handle:
            for number, line in enumerate(handle,1):
                if not line.strip():continue
                try:row=json.loads(line)
                except (ValueError,UnicodeError):raise ValueError(f'{path}:{number}: invalid JSON') from None
                if not isinstance(row,dict):raise ValueError(f'{path}:{number}: expected an object')
                yield str(path),number,row


def prepared_rows(paths, source, max_state, max_pair, workers=0, prefetch=2, chunk_size=64):
    if type(workers) is not int or not 0 <= workers <= 32 or not 1 <= prefetch <= 8 or not 1 <= chunk_size <= 1024:
        raise ValueError('Invalid parallel preparation limits')
    records=source_rows(paths)
    if not workers:
        for path,number,row in records:yield path,number,row,None,None
        return
    def chunks():
        current=[]
        for item in records:
            current.append(item)
            if len(current)==chunk_size:yield current;current=[]
        if current:yield current
    # Bounded submitted tasks and deterministic consumption order. Workers never
    # mutate SQLite, select quotas, choose splits, or publish completed caches.
    pending=deque(); iterator=iter(chunks())
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_codec,initargs=(str(source),max_state,max_pair)) as pool:
        try:
            for _ in range(workers*prefetch):
                chunk=next(iterator,None)
                if chunk is None:break
                pending.append(pool.submit(_encode_chunk,chunk))
            while pending:
                yield from pending.popleft().result()
                chunk=next(iterator,None)
                if chunk is not None:pending.append(pool.submit(_encode_chunk,chunk))
        finally:
            for future in pending:future.cancel()
