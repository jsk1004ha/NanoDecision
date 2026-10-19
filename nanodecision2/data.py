"""Disk-backed token caches and deterministic group-level split auditing."""
from pathlib import Path
import hashlib
import json
import mmap
import sqlite3
import struct
import unicodedata
import zlib
from .schema import normalize_question,gold_target,state_text,TextCodec
from .runtime import sha256

SPLITS=("train","validation","temperature","conformal","test")


def digest(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(group,seed=42):
    v=int(digest(str(seed)+":"+group)[:16],16)/2**64
    for split,upper in zip(SPLITS,(0.70,0.80,0.85,0.90,1.0)):
        if v<upper:return split
    return "test"


def prepare_cache(inputs,out,source_dir,max_state=2048,max_pair=256,seed=42):
    out=Path(out);source=Path(source_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Cache directory must be new/empty")
    out.mkdir(parents=True,exist_ok=True)
    init=json.loads((source/"init.json").read_text(encoding="utf-8"))
    cfg=json.loads((source/"config.json").read_text(encoding="utf-8"))
    if not 1<=max_state<=cfg["max_position_embeddings"] or not 1<=max_pair<=cfg["max_position_embeddings"]:
        raise ValueError("Cache length exceeds encoder maximum")
    codec=TextCodec(source/"tokenizer.json",smoke=init.get("smoke_only",False))
    db=sqlite3.connect(out/"audit.sqlite")
    db.executescript("""CREATE TABLE groups(g TEXT PRIMARY KEY, part TEXT NOT NULL);
        CREATE TABLE states(s TEXT PRIMARY KEY, g TEXT NOT NULL);
        CREATE TABLE examples(signature TEXT PRIMARY KEY, target TEXT NOT NULL, g TEXT NOT NULL);
        CREATE TABLE ids(id TEXT PRIMARY KEY);
    """)
    handles={s:((out/(s+".jsonl")).open("wb"),(out/(s+".idx")).open("wb"),(out/(s+".lengths")).open("wb")) for s in SPLITS}
    counts={s:0 for s in SPLITS};duplicates=0;total_tokens=0;raw_count=0
    try:
        for path in inputs:
            with open(path,encoding="utf-8") as f:
                for line_no,line in enumerate(f,1):
                    if not line.strip():continue
                    raw_count+=1
                    try:
                        row=json.loads(line);group=row.get("group_id")
                        if not isinstance(group,str) or not group:raise ValueError("group_id is required; related variants must share one group")
                        if not isinstance(row.get("id"),str) or not row["id"]:raise ValueError("A unique nonempty id is required")
                        part=row.get("partition") or split_for(group,seed)
                        if part not in SPLITS:raise ValueError("Invalid partition")
                        g=digest(group);old=db.execute("SELECT part FROM groups WHERE g=?",(g,)).fetchone()
                        if old and old[0]!=part:raise ValueError("The same group occurs in two partitions")
                        db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)",(g,part))
                        state=state_text(row["state"])
                        st=digest(unicodedata.normalize("NFKC",state).strip())
                        old=db.execute("SELECT g FROM states WHERE s=?",(st,)).fetchone()
                        if old and old[0]!=g:raise ValueError("Identical normalized state has different group_id values; merge these groups first")
                        db.execute("INSERT OR IGNORE INTO states VALUES (?,?)",(st,g))
                        spec=normalize_question(row);target=gold_target(row,spec)
                        # Detect reordered duplicate options too; preserve labels by description.
                        ordered=sorted((o["description"],float(p),None if spec["values"] is None else spec["values"][i])
                                       for i,(o,p) in enumerate(zip(spec["options"],target)))
                        identity={"state":state,"question":spec["question"],"type":spec["type"],
                                  "options":[(a,c) for a,b,c in ordered]}
                        signature=digest(json.dumps(identity,ensure_ascii=False,sort_keys=True))
                        label=json.dumps([b for a,b,c in ordered])
                        old=db.execute("SELECT target,g FROM examples WHERE signature=?",(signature,)).fetchone()
                        if old:
                            if old[0]!=label or old[1]!=g:raise ValueError("Conflicting duplicate target/group")
                            duplicates+=1;continue
                        db.execute("INSERT INTO examples VALUES (?,?,?)",(signature,label,g))
                        db.execute("INSERT INTO ids VALUES (?)",(row["id"],))
                        sids=codec.state(row["state"],max_state);pairs=codec.pairs(spec,max_pair)
                        weight=float(row.get("weight",1.0))
                        if not 0<weight<=1000000:raise ValueError("Invalid sample weight")
                        encoded={"id":row["id"],"group_id":g,"split":part,"type":spec["type"],
                            "state_ids":sids,"pair_ids":pairs,"target_probs":target,"values":spec["values"],
                            "option_ids":[o["id"] for o in spec["options"]],"weight":weight,
                            "task":row.get("task",spec["type"]),"language":row.get("language","unknown"),
                            "source":row.get("source","unknown"),"reference_option_id":row.get("reference_option_id"),
                            "equivalence_id":row.get("equivalence_id"),
                            "equivalence_verified":row.get("equivalence_verified",False),
                            "label_origin":row.get("label_origin","unspecified")}
                        data,index,lengths=handles[part]
                        index.write(struct.pack("<Q",data.tell()))
                        data.write((json.dumps(encoded,ensure_ascii=False,separators=(",",":"))+"\n").encode())
                        lengths.write(struct.pack("<I",len(sids)+sum(map(len,pairs))))
                        total_tokens+=len(sids)+sum(map(len,pairs));counts[part]+=1
                        if raw_count%10000==0:db.commit()
                    except Exception as e:raise ValueError(f"{path}:{line_no}: {e}") from e
        db.commit()
        group_hash=hashlib.sha256()
        for g,part in db.execute("SELECT g,part FROM groups ORDER BY g"):
            group_hash.update((g+":"+part+"\n").encode())
        group_assignment=group_hash.hexdigest()
    finally:
        for hs in handles.values():
            for h in hs:h.close()
        db.close()
    files={s:{ext:sha256(out/(s+ext)) for ext in (".jsonl",".idx",".lengths")} for s in SPLITS}
    tokenizer_hash="smoke_byte_v1" if init.get("smoke_only") else sha256(source/"tokenizer.json")
    manifest={"version":1,"counts":counts,"raw_rows":raw_count,"duplicates_removed":duplicates,
        "group_assignment_sha256":group_assignment,"input_files_sha256":sorted(sha256(p) for p in inputs),
        "encoded_tokens":total_tokens,"max_state_tokens":max_state,"max_pair_tokens":max_pair,
        "tokenizer_sha256":tokenizer_hash,"files":files,"seed":seed,"smoke_only":init.get("smoke_only",False),
        "split_policy":"group-disjoint 70/10/5/5/10 unless explicitly assigned; exact normalized-state conflicts rejected",
        "limitation":"Near-duplicate, translated and shared-document groups must be supplied correctly by the data author."}
    manifest["fingerprint"]=digest(json.dumps(manifest,sort_keys=True))
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


class CacheDataset:
    def __init__(self,root,split):
        if split not in SPLITS:raise ValueError("Unknown split")
        self.root=Path(root);self.split=split
        self.manifest=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        self.length=int(self.manifest["counts"][split])
        self.encoding=self.manifest.get("cache_encoding","jsonl")
        self.file=(self.root/(split+self.manifest.get("data_suffix",".jsonl"))).open("rb")
        self.index_file=(self.root/(split+".idx")).open("rb")
        self.data=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ) if self.length else None
        self.index=mmap.mmap(self.index_file.fileno(),0,access=mmap.ACCESS_READ) if self.length else None
        if self.length and len(self.index)!=self.length*8:raise ValueError("Corrupt cache index")

    def __len__(self):return self.length

    def __getitem__(self,i):
        if not 0<=i<self.length:raise IndexError(i)
        start=struct.unpack_from("<Q",self.index,8*i)[0]
        end=struct.unpack_from("<Q",self.index,8*(i+1))[0] if i+1<self.length else len(self.data)
        blob=self.data[start:end]
        return json.loads(zlib.decompress(blob) if self.encoding=="zlib_json" else blob)

    def verify(self):
        for suffix,expected in self.manifest["files"][self.split].items():
            if sha256(self.root/(self.split+suffix))!=expected:raise ValueError("Cache changed after preparation")

    def close(self):
        if self.data is not None:self.data.close();self.data=None
        if self.index is not None:self.index.close();self.index=None
        self.file.close();self.index_file.close()

    def __del__(self):
        try:self.close()
        except Exception:pass


def collate(rows,pad_id=0,shuffle=False):
    import random
    import torch
    specs=[]
    for row in rows:
        r=dict(row)
        r["original_order"]=list(range(len(r["pair_ids"])))
        if shuffle and r["type"]!="score":
            order=list(range(len(r["pair_ids"])));random.shuffle(order)
            for field in ("pair_ids","target_probs","option_ids","original_order"):r[field]=[r[field][i] for i in order]
        specs.append(r)
    unique=[];lookup={};state_index=[]
    for r in specs:
        key=tuple(r["state_ids"])
        if key not in lookup:lookup[key]=len(unique);unique.append(r["state_ids"])
        state_index.append(lookup[key])
    b=len(rows);c=max(len(r["pair_ids"]) for r in specs)
    sl=max(map(len,unique));pl=max(len(p) for r in specs for p in r["pair_ids"])
    states=torch.full((len(unique),sl),pad_id,dtype=torch.long);smask=torch.zeros_like(states,dtype=torch.bool)
    pairs=torch.full((b,c,pl),pad_id,dtype=torch.long);pmask=torch.zeros_like(pairs,dtype=torch.bool)
    # Dummy padded candidates need a valid token for SDPA. Their logits are masked.
    pmask[:,:,0]=True
    omask=torch.zeros(b,c,dtype=torch.bool);targets=torch.zeros(b,c);values=torch.zeros(b,c)
    for i,s in enumerate(unique):states[i,:len(s)]=torch.tensor(s);smask[i,:len(s)]=True
    for i,r in enumerate(specs):
        n=len(r["pair_ids"]);omask[i,:n]=True;targets[i,:n]=torch.tensor(r["target_probs"])
        if r["values"] is not None:values[i,:n]=torch.tensor(r["values"])
        for j,p in enumerate(r["pair_ids"]):pairs[i,j,:len(p)]=torch.tensor(p);pmask[i,j,:len(p)]=True
    return {"state_ids":states,"state_mask":smask,"pair_ids":pairs,"pair_mask":pmask,
        "option_mask":omask,"state_index":torch.tensor(state_index),"target":targets,"values":values,
        "ordinal_mask":torch.tensor([r["type"]=="score" for r in specs]),
        "weight":torch.tensor([r["weight"] for r in specs]),
        "original_order":torch.tensor([r["original_order"]+[-1]*(c-len(r["original_order"])) for r in specs])}
