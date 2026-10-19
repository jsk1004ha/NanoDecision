"""Optional continued masked-language pretraining before decision training.

Uses the existing encoder and tied input/output embeddings. A training-only
prediction transform is discarded on export. This is NOT inference-time RAG,
not a new vocabulary, and not a measured claim of Korean improvement.
"""
import json
import math
from pathlib import Path
import sqlite3
import time
from .runtime import sha256
from .schema import TextCodec
from .adapters import fingerprint
from .multilingual import language_key


def prepare_corpus(path,out,init_dir,block_length=256):
    source=Path(init_dir);info=json.loads((source/"init.json").read_text());cfg=json.loads((source/"config.json").read_text())
    if not 8<=block_length<=cfg["max_position_embeddings"]:raise ValueError("Invalid block length")
    out=Path(out)
    if out.exists():raise FileExistsError("Corpus cache exists")
    out.parent.mkdir(parents=True,exist_ok=True)
    codec=TextCodec(source/"tokenizer.json",smoke=info.get("smoke_only",False))
    db=sqlite3.connect(out)
    db.executescript("CREATE TABLE docs(g TEXT PRIMARY KEY,part TEXT); CREATE TABLE texts(t TEXT PRIMARY KEY,g TEXT); CREATE TABLE blocks(id INTEGER PRIMARY KEY, g TEXT,part TEXT,lang TEXT,ids TEXT); CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);")
    counts={};languages={}
    try:
        with open(path,encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line)
                text=row["text"];group=str(row["group_id"]);part=row["partition"];lang=language_key(row["language"])
                if part not in {"train","validation"}:raise ValueError("Only independent pretraining train/validation documents are permitted, not benchmark test/calibration")
                if not isinstance(text,str) or not text.strip():raise ValueError("Empty corpus text")
                old=db.execute("SELECT part FROM docs WHERE g=?",(group,)).fetchone()
                if old and old[0]!=part:raise ValueError("Pretraining document group leaks across splits")
                db.execute("INSERT OR IGNORE INTO docs VALUES (?,?)",(group,part))
                digest=fingerprint(text);old=db.execute("SELECT g FROM texts WHERE t=?",(digest,)).fetchone()
                if old:
                    if old[0]!=group:raise ValueError("Duplicate text has different group IDs; audit first")
                    continue
                db.execute("INSERT INTO texts VALUES (?,?)",(digest,group))
                if codec.smoke:ids=[2]+[b+4 for b in text.encode()]+[1]
                else:ids=codec.tokenizer.encode(text,add_special_tokens=True).ids
                # Keep each window's document-level group for leakage accounting.
                for start in range(0,len(ids),block_length):
                    window=ids[start:start+block_length]
                    if len(window)<4:continue
                    db.execute("INSERT INTO blocks(g,part,lang,ids) VALUES (?,?,?,?)",(group,part,lang,json.dumps(window)))
                    counts[part]=counts.get(part,0)+1;languages[lang]=languages.get(lang,0)+1
        if not counts.get("train") or not counts.get("validation"):raise ValueError("Need nonempty independent corpus train and validation")
        meta={"source_sha256":sha256(path),"counts":counts,"by_language":languages,"tokenizer_sha256":info["tokenizer_sha256"],"block_length":block_length,
              "warning":"Only exact duplicates/group overlap are audited; corpus owners must exclude benchmark/evaluation texts and near-duplicates."}
        db.execute("INSERT INTO meta VALUES (?,?)",("manifest",json.dumps(meta)));db.commit()
    finally:db.close()
    return meta


def mask_tokens(ids,pad_id,mask_id,vocab_size,generator,rate=.15):
    import torch
    valid=ids.ne(pad_id);eligible=valid.clone()
    eligible[:,0]=False
    # Exclude final non-padding token (normally SEP/EOS) as well as padding.
    for i in range(len(ids)):
        last=int(valid[i].sum())-1
        if last>=0:eligible[i,last]=False
    chosen=(torch.rand(ids.shape,generator=generator)<rate)&eligible
    for i in range(len(ids)):
        if not chosen[i].any():
            positions=eligible[i].nonzero().flatten()
            if not len(positions):raise ValueError("Not enough non-special tokens")
            chosen[i,positions[torch.randint(len(positions),(1,),generator=generator)]]=True
    corrupted=ids.clone();u=torch.rand(ids.shape,generator=generator)
    corrupted[chosen & (u<.8)]=mask_id
    random_tokens=torch.randint(vocab_size,ids.shape,generator=generator)
    random_mask=chosen & (u>=.8)&(u<.9);corrupted[random_mask]=random_tokens[random_mask]
    return corrupted,chosen,valid


def tied_mask_loss(hidden,targets,embedding_weight,chunk=32):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    if chunk<1 or not len(targets):raise ValueError("Invalid masked-token chunk")
    def part(h,y):return F.cross_entropy(F.linear(h,embedding_weight).float(),y,reduction="sum")
    losses=[]
    for start in range(0,len(targets),chunk):
        h,y=hidden[start:start+chunk],targets[start:start+chunk]
        losses.append(checkpoint(part,h,y,use_reentrant=False) if torch.is_grad_enabled() else part(h,y))
    return torch.stack(losses).sum()/len(targets)


def adapt(args):
    import torch
    from torch import nn
    from .training import read_model,save_inference_checkpoint
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    if min(args.epochs,args.batch_size,args.mask_chunk,args.threads)<1:raise ValueError("Positive training parameters required")
    if not math.isfinite(args.lr) or args.lr<=0:raise ValueError("Invalid learning rate")
    if args.device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA unavailable")
    if args.bf16 and (args.device!="cuda" or not torch.cuda.is_bf16_supported()):raise ValueError("BF16 requires CUDA BF16 support")
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Adapt into a fresh output directory")
    out.mkdir(parents=True,exist_ok=True)
    source=Path(args.init);info=json.loads((source/"init.json").read_text());model=read_model(source).to(args.device)
    db=sqlite3.connect(f"file:{Path(args.corpus).resolve()}?mode=ro",uri=True)
    meta=json.loads(db.execute("SELECT v FROM meta WHERE k='manifest'").fetchone()[0])
    if meta["tokenizer_sha256"]!=info["tokenizer_sha256"]:raise ValueError("Corpus tokenizer does not match encoder")
    mask_id=args.mask_token_id
    if mask_id is None:
        if info.get("smoke_only"):mask_id=3
        else:
            codec=TextCodec(source/"tokenizer.json")
            for token in ("[MASK]","<mask>","<|mask|>"):
                mask_id=codec.tokenizer.token_to_id(token)
                if mask_id is not None:break
    if mask_id is None or not 0<=mask_id<model.cfg.vocab_size or mask_id==model.cfg.pad_token_id:
        raise ValueError("Pass a valid --mask-token-id from the original tokenizer")
    model.encoder.gradient_checkpointing=args.gradient_checkpointing
    transform=nn.Sequential(nn.Linear(model.cfg.hidden_size,model.cfg.hidden_size),nn.GELU(),nn.LayerNorm(model.cfg.hidden_size)).to(args.device)
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(transform.parameters()),lr=args.lr,weight_decay=.01)
    train_ids=[x[0] for x in db.execute("SELECT id FROM blocks WHERE part='train' ORDER BY id")]
    val_ids=[x[0] for x in db.execute("SELECT id FROM blocks WHERE part='validation' ORDER BY id")]
    def batch(rows):
        tensors=[torch.tensor(json.loads(x[1]),dtype=torch.long) for x in rows]
        return torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True,padding_value=model.cfg.pad_token_id)
    def loss(rows,rng):
        original=batch(rows);corrupt,chosen,valid=mask_tokens(original,model.cfg.pad_token_id,mask_id,model.cfg.vocab_size,rng)
        z=model.encoder(corrupt.to(args.device),valid.to(args.device))[chosen.to(args.device)]
        z=transform(z)
        return tied_mask_loss(z,original[chosen].to(args.device),model.encoder.embedding.weight,args.mask_chunk)
    best=float("inf");history=[];steps=0;start=time.perf_counter()
    rng=torch.Generator().manual_seed(args.seed)
    for ep in range(args.epochs):
        model.train();transform.train();order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(args.seed+ep)).tolist()
        total=0.0;n=0
        for lo in range(0,len(order),args.batch_size):
            rows=[db.execute("SELECT lang,ids FROM blocks WHERE id=?",(train_ids[i],)).fetchone() for i in order[lo:lo+args.batch_size]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(args.device,dtype=torch.bfloat16,enabled=args.bf16):value=loss(rows,rng)
            if not torch.isfinite(value):raise RuntimeError("Non-finite adaptation loss")
            value.backward();torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(transform.parameters()),1,error_if_nonfinite=True);opt.step()
            total+=float(value.detach())*len(rows);n+=len(rows);steps+=1
            if args.max_steps and steps>=args.max_steps:break
        model.eval();transform.eval();vl={}
        with torch.inference_mode():
            for ident in val_ids:
                row=db.execute("SELECT lang,ids FROM blocks WHERE id=?",(ident,)).fetchone()
                # Fixed masks across epochs, independent of training RNG.
                with torch.autocast(args.device,dtype=torch.bfloat16,enabled=args.bf16):v=float(loss([row],torch.Generator().manual_seed(args.seed+100000+ident)))
                cell=vl.setdefault(row[0],[0.0,0]);cell[0]+=v;cell[1]+=1
        val={k:s/c for k,(s,c) in vl.items()};metric=sum(val.values())/len(val)
        entry={"epoch":ep+1,"steps":steps,"train_mask_nll":total/n,"validation_by_language":val,"macro_language_mask_nll":metric}
        history.append(entry);print(json.dumps(entry),flush=True)
        if metric<best:
            best=metric
            new_info={**info,"encoder_adaptation":{"method":"tied-embedding masked LM, no API","corpus":meta,"mask_id":mask_id,"best_validation_macro_nll":best,
               "warning":"No Korean decision accuracy improvement is established until downstream held-out tests."}}
            save_inference_checkpoint(model,out/"best",source,new_info)
            torch.save({"transform":transform.state_dict(),"optimizer":opt.state_dict(),"epoch":ep+1},out/"mlm_training_only.pt")
        if args.max_steps and steps>=args.max_steps:break
    db.close()
    report={"history":history,"wall_seconds":time.perf_counter()-start,"parameter_count_before_and_after":sum(x.numel() for x in model.parameters()),
            "new_runtime_parameters":0,"checkpoint":str(out/"best"),"scope":"Offline fixtures only validated here; full-backbone Korean pretraining is not performed."}
    (out/"adaptation_report.json").write_text(json.dumps(report,indent=2));return report
