"""Strictly proper components; this is supervised learning, NOT RLCD."""
import torch
from torch.nn import functional as F


def proper_loss(logits,target,option_mask,values,ordinal_mask,brier_weight=0.2,rps_weight=0.2):
    if brier_weight<0 or rps_weight<0:raise ValueError("Proper-score weights must be nonnegative")
    z=logits.float().masked_fill(~option_mask,-1e4)
    logp=F.log_softmax(z,dim=-1);p=logp.exp()
    nll=-(target*logp).sum(-1)
    brier=((p-target).square()*option_mask).sum(-1)
    cumulative=(p-target).cumsum(-1)[:,:-1]
    gaps=(values[:,1:]-values[:,:-1]).clamp_min(0)
    valid_gaps=option_mask[:,1:] & option_mask[:,:-1]
    gaps=gaps*valid_gaps
    weights=gaps/gaps.sum(-1,keepdim=True).clamp_min(1e-12)
    rps=(cumulative.square()*weights).sum(-1)*ordinal_mask
    return nll+brier_weight*brier+rps_weight*rps,{"nll":nll,"brier":brier,"rps":rps}
