"""PyTorch training graph and matching C++ graph, with explicit ModernBERT import.

Backbone mathematics follows the public ModernBERT architecture. This is an
independent reference implementation, not a claim to reproduce its fused kernels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig

class EncoderLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        d, f = cfg.hidden_size, cfg.intermediate_size
        self.cfg, self.index = cfg, index
        self.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.mlp_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.mlp_in = nn.Linear(d, 2*f, bias=False)
        self.mlp_out = nn.Linear(f, d, bias=False)

    def forward(self, x, valid):
        b, length, d = x.shape
        cfg = self.cfg
        heads = cfg.num_attention_heads
        hd = d // heads
        q, k, v = self.qkv(self.attn_norm(x)).view(b, length, 3, heads, hd).permute(2, 0, 3, 1, 4).unbind(0)
        local = self.index % cfg.global_attn_every_n_layers != 0
        theta = cfg.local_rope_theta if local else cfg.global_rope_theta
        freq = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd)
        phase = torch.arange(length, dtype=torch.float32, device=x.device)[:, None] * freq[None]
        phase = torch.cat([phase, phase], -1)
        cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)
        def rotate(a):
            return a*cos + torch.cat([-a[..., hd//2:], a[..., :hd//2]], -1)*sin
        q, k = rotate(q), rotate(k)
        if not local:
            mask = valid[:, None, None, :]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # Chunked sliding-window attention: do not allocate a full L x L mask.
            radius = cfg.local_attention // 2
            parts = []
            for lo in range(0, length, 128):
                hi = min(length, lo+128)
                kl, kr = max(0, lo-radius), min(length, hi+radius)
                pos_q = torch.arange(lo, hi, device=x.device)[:, None]
                pos_k = torch.arange(kl, kr, device=x.device)[None]
                mask = (pos_q-pos_k).abs() <= radius
                mask = mask[None, None] & valid[:, None, None, kl:kr]
                parts.append(F.scaled_dot_product_attention(q[:, :, lo:hi], k[:, :, kl:kr],
                                                           v[:, :, kl:kr], attn_mask=mask))
            a = torch.cat(parts, dim=2)
        x = x + self.attn_out(a.transpose(1, 2).reshape(b, length, d))
        act, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.gelu(act, approximate="none") * gate)

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.embed_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.layers = nn.ModuleList([EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.gradient_checkpointing = False

    def forward(self, ids, valid):
        x = self.embed_norm(self.embedding(ids))
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(layer, x, valid, use_reentrant=False)
            else:
                x = layer(x, valid)
        return self.final_norm(x)

class DecisionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.hidden_size, cfg.decision_dim
        self.key = nn.Linear(d, h, bias=False)
        self.value = nn.Linear(d, h, bias=False)
        self.candidate = nn.Linear(d, h, bias=False)
        self.updates = nn.ModuleList([nn.Linear(2*h, h) for _ in range(cfg.decision_hops)])
        self.norms = nn.ModuleList([nn.LayerNorm(h, eps=cfg.norm_eps) for _ in range(cfg.decision_hops)])
        self.local = nn.Linear(4*h, h)
        self.set_layer = nn.Linear(3*h, h)
        self.output = nn.Linear(h, 1)
        self.dropout = nn.Dropout(0.1)
        self.use_set_context = True

    def forward(self, keys, values, context_mask, candidate_vecs, option_mask):
        q = candidate_vecs
        for update, norm in zip(self.updates, self.norms):
            scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(~context_mask[:, None, :], -1e4)
            att = torch.matmul(scores.softmax(-1), values)
            q = norm(q + F.gelu(update(torch.cat([q, att], dim=-1)), approximate="none"))
        u = F.gelu(self.local(torch.cat([q, candidate_vecs, q*candidate_vecs,
                                         (q-candidate_vecs).abs()], dim=-1)), approximate="none")
        u = self.dropout(u)
        if self.use_set_context:
            mean = (u*option_mask[..., None]).sum(1, keepdim=True) / option_mask.sum(1)[:, None, None].clamp_min(1)
            maximum = u.masked_fill(~option_mask[..., None], -1e4).amax(1, keepdim=True)
        else:
            mean = torch.zeros_like(u[:, :1]); maximum = mean
        # Permutation-equivariant competition, not class-index embeddings.
        features = torch.cat([u, mean.expand_as(u), maximum.expand_as(u)], dim=-1)
        logits = self.output(F.gelu(self.set_layer(features), approximate="none")).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e4)

class DecisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.head = DecisionHead(cfg)

    def forward(self, state_ids, state_mask, pair_ids, pair_mask, option_mask, state_index=None):
        # state_ids can contain U unique states for B decision questions.
        context = self.encoder(state_ids, state_mask)
        keys, values = self.head.key(context), self.head.value(context)
        if state_index is not None:
            keys, values = keys[state_index], values[state_index]
            state_mask = state_mask[state_index]
        b, c, length = pair_ids.shape
        hidden = self.encoder(pair_ids.reshape(b*c, length), pair_mask.reshape(b*c, length))
        m = pair_mask.reshape(b*c, length)[..., None]
        pooled = (hidden*m).sum(1) / m.sum(1).clamp_min(1)
        candidates = self.head.candidate(pooled).view(b, c, -1)
        return self.head(keys, values, state_mask, candidates, option_mask)

    def load_hf_encoder(self, state):
        target = {}
        mapping = {
            "embedding.weight": "embeddings.tok_embeddings.weight",
            "embed_norm.weight": "embeddings.norm.weight",
            "final_norm.weight": "final_norm.weight",
        }
        for i in range(self.cfg.num_hidden_layers):
            prefix = f"layers.{i}."
            for ours, theirs in [("qkv.weight", "attn.Wqkv.weight"),
                                 ("attn_out.weight", "attn.Wo.weight"),
                                 ("mlp_norm.weight", "mlp_norm.weight"),
                                 ("mlp_in.weight", "mlp.Wi.weight"),
                                 ("mlp_out.weight", "mlp.Wo.weight")]:
                mapping[prefix+ours] = prefix+theirs
            if i:
                mapping[prefix+"attn_norm.weight"] = prefix+"attn_norm.weight"
        for ours, original in mapping.items():
            keys = [original, "model."+original]
            present = [k for k in keys if k in state]
            if len(present) != 1:
                raise ValueError(f"Missing or ambiguous backbone weight {original}")
            target[ours] = state[present[0]]
        self.encoder.load_state_dict(target, strict=True)
