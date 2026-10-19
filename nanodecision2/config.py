from dataclasses import dataclass, asdict
import math

@dataclass
class ModelConfig:
    vocab_size: int = 256000
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    max_position_embeddings: int = 8192
    decision_dim: int = 256
    decision_hops: int = 2
    pad_token_id: int = 0
    norm_eps: float = 1e-5
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 160000.0

    def __post_init__(self):
        limits = {"vocab_size": (8, 1000000), "hidden_size": (8, 2048),
            "intermediate_size": (8, 8192), "num_hidden_layers": (1, 64),
            "num_attention_heads": (1, 64), "global_attn_every_n_layers": (1, 64),
            "local_attention": (2, 8192), "max_position_embeddings": (8, 32768),
            "decision_dim": (8, 2048), "decision_hops": (1, 8)}
        for key, (lo, hi) in limits.items():
            v = getattr(self, key)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("Attention head dimension must be an even integer")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("Invalid pad token")
        for k in ("norm_eps", "global_rope_theta", "local_rope_theta"):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f"Invalid {k}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def smoke(cls):
        return cls(vocab_size=260, hidden_size=32, intermediate_size=48,
            num_hidden_layers=3, num_attention_heads=4, global_attn_every_n_layers=2,
            local_attention=8, max_position_embeddings=512, decision_dim=24, decision_hops=2)

    @classmethod
    def from_hf(cls, d, decision_dim=256, decision_hops=2):
        # Supported explicitly; reject silent architectural mismatches.
        if d.get("model_type") != "modernbert":
            raise ValueError("Only ModernBERT/mmBERT import is supported")
        for field in ("attention_bias", "mlp_bias", "norm_bias"):
            if d.get(field, False):
                raise ValueError(f"Unsupported backbone: {field}=true")
        if d.get("hidden_activation", "gelu") != "gelu" or d.get("rope_scaling"):
            raise ValueError("Only exact GELU and standard full rotary embeddings are supported")
        fields = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "global_attn_every_n_layers", "local_attention",
            "max_position_embeddings", "pad_token_id", "norm_eps", "global_rope_theta"]
        values = {k: d[k] for k in fields}
        values["local_rope_theta"] = d.get("local_rope_theta") or d["global_rope_theta"]
        return cls(**values, decision_dim=decision_dim, decision_hops=decision_hops)
