"""Export to mmap-native format; all matrix weights can stay INT8 at runtime."""
import json
import os
from pathlib import Path
import shutil
import struct
import numpy as np
from .runtime import sha256


def write_native(model,path,quantize=True):
    if not model.head.use_set_context:
        raise ValueError("Set-context ablation must use the PyTorch evaluator; native graph exports full head only")
    cfg=model.cfg;items=list(model.state_dict().items())
    with open(path,"wb") as f:
        f.write(b"NDC2BIN\0");f.write(struct.pack("<I",1))
        fields=["vocab_size","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads",
                "global_attn_every_n_layers","local_attention","max_position_embeddings","decision_dim","decision_hops","pad_token_id"]
        f.write(struct.pack("<11I",*(getattr(cfg,k) for k in fields)))
        f.write(struct.pack("<3f",cfg.norm_eps,cfg.global_rope_theta,cfg.local_rope_theta))
        f.write(struct.pack("<I",len(items)))
        for name,tensor in items:
            a=tensor.detach().float().cpu().numpy().astype("<f4",copy=False)
            if a.ndim not in (1,2) or not np.isfinite(a).all():raise ValueError("Invalid export tensor "+name)
            rows=a.shape[0];cols=a.shape[1] if a.ndim==2 else 1
            quant=bool(quantize and a.ndim==2)
            if quant:
                scales=np.maximum(np.max(np.abs(a),axis=1)/127,1e-12).astype("<f4")
                q=np.clip(np.rint(a/scales[:,None]),-127,127).astype("i1")
                payload=scales.tobytes()+q.tobytes()
            else:payload=a.tobytes()
            raw=name.encode();f.write(struct.pack("<I",len(raw)));f.write(raw)
            f.write(struct.pack("<4IQ",a.ndim,rows,cols,int(quant),len(payload)))
            f.write(b"\0"*((-f.tell())%64));f.write(payload)


def export_bundle(model,out,source_dir=None,quantize=True,smoke_only=False,
                  max_state_tokens=None,max_pair_tokens=256,provenance=None):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Output must be a new/empty directory")
    out.mkdir(parents=True,exist_ok=True)
    max_state_tokens=max_state_tokens or model.cfg.max_position_embeddings
    if not 1<=max_state_tokens<=model.cfg.max_position_embeddings or not 1<=max_pair_tokens<=model.cfg.max_position_embeddings:
        raise ValueError("Runtime limits must fit the trained configuration")
    write_native(model,out/"model.nd2",quantize)
    if not smoke_only:
        if source_dir is None or not (Path(source_dir)/"tokenizer.json").exists():raise ValueError("Missing trained tokenizer.json")
        shutil.copyfile(Path(source_dir)/"tokenizer.json",out/"tokenizer.json")
    manifest={"format":"nanodecision2-native","version":1,"config":model.cfg.to_dict(),
        "model_sha256":sha256(out/"model.nd2"),"matrix_storage":"int8_rowwise" if quantize else "float32",
        "arithmetic":"float32 (optional dequantized BLAS tiles)","smoke_only":bool(smoke_only),
        "max_state_tokens":max_state_tokens,"max_pair_tokens":max_pair_tokens,
        "provenance":provenance or {},"quality_status":"UNVERIFIED: run task evaluation; file export is not a quality result"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
