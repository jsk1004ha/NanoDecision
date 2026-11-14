"""Deterministic, padding-aware batches and equal-step distributed assignment.

Budgets count padded encoder tokens, not FLOPs. No sample is silently dropped
or counted twice. Empty final ranks are explicit zero-weight work, not data.
"""
from itertools import islice
import json
from pathlib import Path
import numpy as np
from .auto_sources import atomic_json


def cache_shapes(dataset):
    """Disk-backed (state length, candidate count, maximum pair length) table."""
    view_hash = __import__('hashlib').sha256(np.asarray(dataset.indices,dtype='<i8').tobytes()).hexdigest() if hasattr(dataset,'indices') else 'all'
    path = Path(dataset.root) / (dataset.split + '.' + view_hash[:12] + '.budget-shapes.npy')
    lock = path.with_suffix('.json')
    identity = {'fingerprint': dataset.manifest['fingerprint'], 'rows': len(dataset), 'version': 1, 'view_hash': view_hash}
    if path.exists() and lock.exists() and json.loads(lock.read_text()) == identity:
        shapes = np.load(path, mmap_mode='r', allow_pickle=False)
        if shapes.shape == (len(dataset), 3) and shapes.dtype == np.dtype('uint32'):
            return shapes
        raise ValueError('Corrupt batch-shape index; rebuild it')
    temporary = path.with_name(path.name + '.partial')
    shapes = np.lib.format.open_memmap(temporary, mode='w+', dtype='uint32', shape=(len(dataset), 3))
    for i in range(len(dataset)):
        row = dataset[i]
        shapes[i] = len(row['state_ids']), len(row['pair_ids']), max(map(len, row['pair_ids']))
    shapes.flush(); del shapes
    temporary.replace(path); atomic_json(lock, identity)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def padded_cost(shapes, ids):
    if not ids:
        return 0
    maximum = np.asarray(shapes)[ids].max(axis=0).astype(np.int64)
    return len(ids) * int(maximum[0] + maximum[1] * maximum[2])


def token_batches(shapes, max_examples, max_tokens=0, seed=42, epoch=0, units=None):
    """Group within shuffled pools; an approved pair stays in one micro-batch."""
    if type(max_examples) is not int or max_examples < 1 or type(max_tokens) is not int or max_tokens < 0:
        raise ValueError('Positive max_examples and nonnegative max_tokens required')
    shapes = np.asarray(shapes)
    if shapes.ndim != 2 or shapes.shape[1] != 3 or np.any(shapes < 1):
        raise ValueError('Expected positive state/candidate/pair shapes')
    count = len(shapes) if units is None else len(units)
    order = np.arange(count, dtype=np.int64)
    np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(order)
    scores = shapes[:, 0].astype(np.int64) + shapes[:, 1].astype(np.int64) * shapes[:, 2]
    for start in range(0, count, 2048):
        part = order[start:start + 2048]
        costs = scores[part] if units is None else [max(scores[list(units[int(j)])]) for j in part]
        order[start:start + len(part)] = part[np.argsort(costs, kind='stable')]
    batch = []
    for item in order:
        unit = [int(item)] if units is None else list(units[int(item)])
        if not unit or len(unit) > max_examples or (max_tokens and padded_cost(shapes, unit) > max_tokens):
            raise ValueError('A sample/pair exceeds the declared batch budget; no truncation performed')
        combined = batch + unit
        if batch and (len(combined) > max_examples or (max_tokens and padded_cost(shapes, combined) > max_tokens)):
            yield batch
            batch = []
        batch.extend(unit)
    if batch:
        yield batch


def distributed_rounds(batches, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError('world_size must be positive')
    iterator = iter(batches)
    while True:
        row = list(islice(iterator, world_size))
        if not row:
            return
        yield row + [[] for _ in range(world_size - len(row))]


def update_groups(rounds, accumulation):
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('accumulation must be positive')
    iterator = iter(rounds)
    while group := list(islice(iterator, accumulation)):
        yield group
