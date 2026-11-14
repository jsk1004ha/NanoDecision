"""torchrun lifecycle and exact global weighted-gradient normalization."""
from contextlib import nullcontext
from datetime import timedelta
import os
import random
import torch
import torch.distributed as dist


class DistributedRun:
    def __init__(self, enabled=False, device='cpu'):
        self.world = int(os.environ.get('WORLD_SIZE', '1')) if enabled else 1
        if not enabled and int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise ValueError('torchrun requires --ddp; refusing concurrent independent writers')
        self.rank = int(os.environ.get('RANK', '0')) if enabled else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0')) if enabled else 0
        self.owned = False
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cpu')
        if self.world > 1:
            if dist.is_initialized():
                raise ValueError('Use an isolated torchrun process for this trainer')
            dist.init_process_group('nccl' if device == 'cuda' else 'gloo', timeout=timedelta(minutes=30))
            self.owned = True

    def close(self):
        if self.owned:
            dist.destroy_process_group(); self.owned = False

    def primary(self, fn):
        """Broadcast rank-zero errors before any rank can enter another collective."""
        message = [None]
        if self.rank == 0:
            try:
                message[0] = {'result': fn()}
            except Exception as exc:
                message[0] = {'error': type(exc).__name__ + ': ' + str(exc)}
        if self.world > 1:
            dist.broadcast_object_list(message, src=0)
        if 'error' in message[0]:
            raise RuntimeError(message[0]['error'])
        return message[0]['result']

    def sync_context(self, wrapped, last):
        return wrapped.no_sync() if self.world > 1 and not last else nullcontext()

    def sum(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor.item()

    def scale_gradients(self, model, local_weight):
        total = self.sum(local_weight)
        if not total > 0:
            raise RuntimeError('Empty weighted update')
        # DDP already averaged gradients across ranks; undo that before /global S.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(self.world / total)
        return total

    def rng_states(self):
        state = {'python': random.getstate(), 'torch': torch.get_rng_state(),
                 'cuda': torch.cuda.get_rng_state() if self.device.type == 'cuda' else None}
        states = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        return states

    def restore_rng(self, states):
        if len(states) != self.world:
            raise ValueError('Exact resume requires the same world size')
        state = states[self.rank]
        random.setstate(state['python']); torch.set_rng_state(state['torch'])
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda'])
