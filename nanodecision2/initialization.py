import json
from pathlib import Path
import shutil
from .runtime import sha256


def initialize(out,backbone="jhu-clsp/mmBERT-base",revision="main",smoke=False,skip_upstream_check=False,decision_dim=256):
    import torch
    from safetensors.torch import load_file
    from .config import ModelConfig
    from .model import DecisionModel
    from .training import save_inference_checkpoint
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Initialize into a new/empty directory")
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    if smoke:
        model=DecisionModel(ModelConfig.smoke())
        info={"smoke_only":True,"backbone":"random offline fixture","tokenizer_sha256":"smoke_byte_v1",
              "upstream_parity":"not applicable","warning":"Not a useful pretrained decision model"}
    else:
        from huggingface_hub import snapshot_download
        snapshot=Path(snapshot_download(backbone,revision=revision,allow_patterns=["config.json","tokenizer.json",
                    "tokenizer_config.json","special_tokens_map.json","*.safetensors","*.safetensors.index.json"]))
        hf_config=json.loads((snapshot/"config.json").read_text())
        cfg=ModelConfig.from_hf(hf_config,decision_dim=decision_dim)
        model=DecisionModel(cfg)
        weights={}
        for shard in sorted(snapshot.glob("*.safetensors")):
            part=load_file(str(shard),device="cpu")
            if weights.keys() & part.keys():raise ValueError("Duplicate keys across safetensor shards")
            weights.update(part)
        if not weights:raise ValueError("No safetensors weights downloaded")
        model.load_hf_encoder(weights);del weights
        shutil.copyfile(snapshot/"tokenizer.json",out/"tokenizer.json")
        info={"smoke_only":False,"backbone":backbone,"requested_revision":revision,
              "resolved_revision":snapshot.name,"tokenizer_sha256":sha256(out/"tokenizer.json"),
              "upstream_parity":"NOT_RUN"}
        if not skip_upstream_check:
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise RuntimeError("Install '.[verify]' for the mandatory upstream parity check, or explicitly use --skip-upstream-check") from e
            original=AutoModel.from_pretrained(str(snapshot),attn_implementation="sdpa",reference_compile=False,
                                               torch_dtype=torch.float32,local_files_only=True).eval()
            model.eval();generator=torch.Generator().manual_seed(20260928);errors=[]
            with torch.inference_mode():
                for length in (9,137):
                    ids=torch.randint(5,cfg.vocab_size,(1,length),generator=generator)
                    mask=torch.ones_like(ids,dtype=torch.bool)
                    a=model.encoder(ids,mask);b=original(input_ids=ids,attention_mask=mask).last_hidden_state
                    error=float((a-b).abs().max());errors.append(error)
                    if not torch.allclose(a,b,atol=2e-4,rtol=2e-4):
                        raise RuntimeError(f"Imported encoder differs from upstream, max error {error}; initialization blocked")
            del original
            info["upstream_parity"]={"max_abs_errors":errors,"lengths":[9,137],"device":"cpu_fp32"}
    save_inference_checkpoint(model,out,out,info)
    return info
