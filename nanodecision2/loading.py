"""Parallel cache decompression without worker-dependent shuffle/RNG changes.

Workers read raw cached rows; the training process still applies weights,
option shuffling, and collate. Checkpoints store consumed batch indexes, never
prefetch positions. This improves I/O overlap, not neural batch size by itself.
"""
import math


def _identity(value):
    return value


def _worker_init(_worker_id):
    import torch
    torch.set_num_threads(1)


class RowBatches:
    def __init__(self, dataset, order, batch_size, start=0, paired=None):
        self.dataset, self.order = dataset, order
        self.batch_size, self.start, self.paired = batch_size, start, paired
        self.total = len(paired) if paired is not None else math.ceil(len(order)/batch_size)
        if not 0 <= start <= self.total:
            raise ValueError('Consumed batch cursor is out of range')

    def __len__(self):
        return self.total - self.start

    def __getitem__(self, offset):
        if not 0 <= offset < len(self):
            raise IndexError(offset)
        index = self.start + offset
        ids = self.paired[index] if self.paired is not None else self.order[index*self.batch_size:(index+1)*self.batch_size]
        return index, [self.dataset[int(i)] for i in ids]


def row_batches(dataset, order, batch_size, start=0, paired=None, workers=0,
                prefetch=2, seed=0):
    if type(workers) is not int or not 0 <= workers <= 64:
        raise ValueError('workers must be 0..64')
    if type(prefetch) is not int or not 1 <= prefetch <= 8:
        raise ValueError('prefetch must be 1..8')
    batches = RowBatches(dataset, order, batch_size, start, paired)
    if workers == 0:
        for i in range(len(batches)):
            yield batches[i]
        return
    import torch
    from torch.utils.data import DataLoader
    # Isolate DataLoader's base-seed draw from the model/dropout RNG.
    loader = DataLoader(batches, batch_size=None, num_workers=workers,
                        multiprocessing_context='spawn', prefetch_factor=prefetch,
                        worker_init_fn=_worker_init, collate_fn=_identity,
                        generator=torch.Generator().manual_seed(seed))
    yield from loader
