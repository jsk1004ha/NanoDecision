"""Launched by the distributed integration test; not a performance benchmark."""
import json
from pathlib import Path
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
from nanodecision2.distributed_support import DistributedRun

run=DistributedRun(True,'cpu')
try:
    torch.set_num_threads(1);torch.manual_seed(7)
    model=torch.nn.Linear(4,3).double()
    reference=torch.nn.Linear(4,3).double();reference.load_state_dict(model.state_dict())
    wrapped=DistributedDataParallel(model,broadcast_buffers=False)
    x=torch.arange(20,dtype=torch.float64).reshape(5,4)/7
    y=torch.arange(15,dtype=torch.float64).reshape(5,3)/11
    weights=torch.arange(1,6,dtype=torch.float64)
    ((reference(x)-y).square().sum(-1)*weights).sum().div(weights.sum()).backward()
    local=[[0],[2]] if run.rank==0 else [[1],[3,4]]
    local_weight=0.
    for k,ids in enumerate(local):
        with run.sync_context(wrapped,k==1):
            loss=((wrapped(x[ids])-y[ids]).square().sum(-1)*weights[ids]).sum();loss.backward()
        local_weight+=float(weights[ids].sum())
    total=run.scale_gradients(model,local_weight)
    error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),reference.parameters()))
    if error>1e-12 or total!=15.:raise RuntimeError('Global weighted gradient mismatch')
    run.primary(lambda:Path(sys.argv[1]).write_text(json.dumps({'max_abs_gradient_error':error,'global_weight':total,'world_size':run.world})))
finally:run.close()
