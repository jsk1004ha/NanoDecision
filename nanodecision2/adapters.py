"""Pure record adapters, usable with HF streams or local JSONL/TSV releases.

Preserve official split names in provenance. Training-source originals may be
split further into train/validation/temperature/conformal using group hashes;
official test remains test. Do not infer that evaluation sources trained Laya.
"""
import hashlib
import json
import unicodedata
from .schema import normalize_question, gold_target
from .multilingual import language_key

NLI_KO = {"entailment":"전제로부터 가설이 참임을 알 수 있다.",
          "neutral":"전제만으로 가설의 참과 거짓을 결정할 수 없다.",
          "contradiction":"전제에 따르면 가설은 거짓이다."}
NLI_EN = {"entailment":"The premise entails the hypothesis.",
          "neutral":"The premise does not determine whether the hypothesis is true.",
          "contradiction":"The premise contradicts the hypothesis."}
SCENARIOS_KO = {"alarm":"알람 설정과 관리", "audio":"음량 등 오디오 제어", "calendar":"일정과 달력 관리",
 "cooking":"요리와 조리 시간", "datetime":"날짜와 시간 확인", "email":"이메일 작성·검색·관리",
 "general":"인사·일반 대화", "iot":"스마트홈 기기 제어", "lists":"목록 작성과 관리", "music":"음악 재생과 검색",
 "news":"뉴스 요청", "play":"라디오·팟캐스트·오디오북 재생", "qa":"지식에 관한 질문과 답변", "recommendation":"장소 등 추천 요청",
 "social":"소셜 미디어 활동", "takeaway":"음식 포장·배달 주문", "transport":"교통·이동 정보", "weather":"날씨 정보"}
YNAT_KO = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
QUEUE_EN = {"Technical Support":"technical faults, bugs and integrations", "Product Support":"help using product features",
 "Customer Service":"general customer account and service questions", "IT Support":"internal IT devices and access",
 "Billing and Payments":"invoices, charges and payments", "Returns and Exchanges":"returning or exchanging products",
 "Service Outages and Maintenance":"downtime and scheduled maintenance", "Sales and Pre-Sales":"buying, pricing and quotes",
 "Human Resources":"employment and personnel matters", "General Inquiry":"other general inquiries"}


def fingerprint(text):
    return hashlib.sha256(unicodedata.normalize("NFC", str(text)).strip().encode()).hexdigest()


def partition_for(group, source_split):
    if source_split == "test": return "test"
    v = int(fingerprint(group)[:12], 16) / 16**12
    if source_split in {"validation", "dev", "validation_matched", "validation_mismatched"}:
        # Official development examples are NEVER turned into test evidence.
        return "validation"
    if source_split != "train": raise ValueError("Unsupported original split")
    return "train" if v < .85 else "validation" if v < .90 else "temperature" if v < .95 else "conformal"


def base_row(source, source_split, original_id, group, language):
    return {"id":f"{source}:{source_split}:{language}:{original_id}", "group_id":group,
            "source":source,"source_split":source_split,"partition":partition_for(group,source_split),
            "language":language,"label_origin":"dataset_label"}


def nli_row(r, source, source_split="train", language="ko", label_names=None, original_id=None, group=None):
    if r.get("label", -1) in (-1, "-", None): return []
    label=r["label"]
    if isinstance(label,int):
        if not label_names: raise ValueError("Integer NLI labels require the source's feature names")
        label=label_names[label]
    if label not in NLI_EN: raise ValueError("Unknown NLI label")
    # Default premise grouping is conservative. For translated NLI use an
    # explicit shared origin ID from the bilingual alignment, not a guessed hash.
    group=group or "nli-premise:"+fingerprint(r["premise"])
    row=base_row(source,source_split,original_id or r.get("guid") or r.get("pairID") or fingerprint(str(r)),group,language)
    ko=language_key(language)=="ko"
    row.update(type="choice",state={"premise":r["premise"],"hypothesis":r["hypothesis"]},
        question="전제와 가설의 관계는 무엇인가?" if ko else "What is the relation between the premise and hypothesis?",
        criteria=NLI_KO if ko else NLI_EN,target=label,task="nli")
    return [row]


def paws_row(r, source_split="train", language="ko"):
    # PAWS-X id corresponds to the source pair within the original split.
    if int(r["label"]) not in (0,1):raise ValueError("PAWS-X label must be 0 or 1")
    group=f"paws-x:{source_split}:{r['id']}"
    row=base_row("paws-x",source_split,r["id"],group,language)
    ko=language_key(language)=="ko"
    row.update(type="noul",state={"sentence1":r["sentence1"],"sentence2":r["sentence2"]},
        question="두 문장은 같은 의미를 전달하는가?" if ko else "Do the two sentences convey the same meaning?",
        criteria={"false":"의미가 다르다.","true":"의미가 같다."} if ko else {"false":"Different meanings.","true":"Equivalent meanings."},
        target=bool(int(r["label"])),task="paraphrase")
    row["label_origin"] = "machine_translated_train" if source_split=="train" and language_key(language)!="en" else "dataset_label"
    return [row]


def massive_row(r, task="scenario", labels=None, prompt_language=None):
    split=r["partition"];lang=r["locale"];group=f"massive:{r['id']}"
    task = task.replace("massive-", "")
    if task not in {"scenario","intent"}:raise ValueError("Unknown MASSIVE task")
    ko=language_key(prompt_language or lang)=="ko"
    if task=="scenario":
        labels=list(SCENARIOS_KO)
        criteria=SCENARIOS_KO if ko else {x:x.replace("_"," ") for x in labels}
    else:
        if not labels or len(set(labels))!=60:raise ValueError("Full-intent evaluation requires the official 60-label inventory; no gold-assisted top-k")
        criteria={x:x.replace("_"," ") for x in labels}
        # Intents are English descriptions unless a reviewed description mapping is supplied separately.
    target=r[task]
    if target not in criteria:raise ValueError("Unknown MASSIVE label")
    row=base_row("massive",split,str(r["id"])+":"+task,group,lang)
    row.update(type="choice",state=r["utt"])
    row["question"]=("요청은 어떤 상황에 해당하는가?" if ko and task=="scenario" else
                     "What scenario does this request belong to?" if task=="scenario" else
                     "요청의 구체적인 의도는 무엇인가?" if ko else "What is the specific intent of the request?")
    row.update(criteria=criteria,target=target,task="massive_"+task,
               label_origin="human_localized_parallel",prompt_language="ko" if ko else "en",
               equivalence_id=f"{group}:{task}",equivalence_verified=False)
    return [row]


def typed_rows(r, source_split="train"):
    """Keep teacher probabilities AND the dataset's separate published label.

    Do not convert teacher distributions into human-correctness labels. The
    original JSON state is passed verbatim in meaning, never the latent facts.
    """
    def obj(x):return json.loads(x) if isinstance(x,str) else x
    state,questions,gold=(obj(r[k]) for k in ("state","questions","gold"))
    original=r.get("id") or r.get("case_id") or fingerprint(json.dumps(state,ensure_ascii=False,sort_keys=True))
    group="typed-decisions:"+str(original);out=[]
    for name,q in questions.items():
        q=dict(q);kind=q["type"]
        criteria=q.get("criteria")
        if kind=="choice":q["criteria"]={str(k):v if isinstance(v,str) and v else str(k) for k,v in criteria.items()}
        elif kind=="noul" and not criteria:
            q["criteria"]={"false":"false","true":"true"}
        elif kind=="score":
            # JSON benchmark score criteria are ordered descriptions.
            if not isinstance(criteria,list):raise ValueError("Typed score criteria must be an ordered list")
            q["criteria"]=[x if isinstance(x,str) else json.dumps(x,ensure_ascii=False) for x in criteria]
        row=base_row("typed-decisions",source_split,f"{original}:{name}",group,"en")
        row.update(q);row.update(state=state,task=f"typed:{r.get('workflow','unknown')}:{name}",label_origin="teacher_distribution_synthetic")
        spec=normalize_question(row);ids=[o["id"] for o in spec["options"]];g=gold[name]
        probs=g.get("probabilities")
        if not isinstance(probs,dict):raise ValueError("Typed dataset gold needs an explicit probability mapping")
        probs={str(k).lower() if isinstance(k,bool) else str(k):float(v) for k,v in probs.items()}
        if set(probs)!=set(ids):raise ValueError("Typed gold candidate keys do not match schema")
        row["target_probs"]=[probs[k] for k in ids]
        label=g.get("label")
        label=str(label).lower() if isinstance(label,bool) else str(label)
        if label not in ids:raise ValueError("Typed published gold label is not an option")
        row["reference_option_id"]=label
        # Teacher-labeled data is allowed for fit/validation, but not for a
        # claim of conformal coverage of human-correct labels.
        if row["partition"]=="conformal":row["partition"]="validation"
        gold_target(row,spec);out.append(row)
    return out


def simple_source_rows(r, source, source_split="train", original_id=None):
    original_id=original_id or fingerprint(json.dumps(r,ensure_ascii=False,sort_keys=True))
    text=r.get("text", "")
    row=base_row(source,source_split,original_id,source+":"+fingerprint(str(original_id)),"en")
    if source=="klue-ynat":
        row.update(language="ko",state=r["title"],type="choice",question="이 뉴스 제목의 주제는 무엇인가?",
                   criteria={str(i):x for i,x in enumerate(YNAT_KO)},target=str(r["label"]),task="topic")
    elif source=="ag-news":
        row.update(state=text,type="choice",question="What is the article's topic?",
                   criteria=dict(zip(["0","1","2","3"],["World news","Sports","Business","Science and technology"])),target=str(r["label"]),task="topic")
    elif source=="enron-spam":
        row.update(state={"subject":r.get("subject") or "", "message":r.get("message") or text},type="noul",
                   question="Is this unsolicited spam or bulk marketing?",target=bool(int(r["label"])),task="spam")
    elif source=="phishing":
        label=r.get("Email Type")
        if label not in {"Safe Email","Phishing Email"} or not r.get("Email Text"):return []
        row.update(state=r["Email Text"],type="noul",question="Is this email a phishing attempt?",target=label=="Phishing Email",task="phishing")
    elif source=="support-queue":
        if r.get("language")!="en" or r.get("queue") not in QUEUE_EN or not r.get("body"):return []
        row.update(state={"subject":r.get("subject") or "", "body":r["body"]},type="choice",
                   question="Which support queue should handle this ticket?",criteria=QUEUE_EN,target=r["queue"],
                   task="support_queue",label_origin="dataset_label_provenance_requires_review")
    else:raise ValueError("Unknown source")
    gold_target(row,normalize_question(row));return [row]
