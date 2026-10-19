# NanoDecision2 0.3.0 — 한국어 학습 강화 / Julia·Laya 비교

2026-09-29. Jev API가 필요 없는 구조화 확률 판단 모델의 학습·검증·C++ 실행 코드입니다.
**완성된 한국어 본학습 가중치나 Julia/Laya를 능가한 실측 결과는 포함하지 않습니다.**
이번 배포는 v0.2.0을 실제 수정한 소스입니다. 새 기능은 작은 오프라인 fixture로 검증했고,
실제 mmBERT 가중치·공개 데이터 전체·Julia/Laya 가중치·CUDA 경로는 이 제작 환경에서 실행하지 못했습니다.

## 1. 이번에 달라진 점

| 부분 | 구현 |
|---|---|
| 인코더 추가학습 | 기존 mmBERT 어휘·구조를 유지한 masked-language-model 추가학습. 출력 임베딩을 입력 임베딩에 묶고 마스킹 위치만 작은 묶음으로 계산. 추가학습용 transform은 배포에서 제거 |
| 한국어 데이터 | KorNLI, KLUE-NLI/YNAT, PAWS-X 한국어, MASSIVE 18상황/60의도 변환. 원본 분할·언어·정답 출처·원본 그룹 보존 |
| 언어별 학습 | 언어별 loss weight와 노출량 기록, 언어별 검증 NLL/정확도, macro-language NLL 모델 선택 |
| 한·영 대응 학습 | 검토된 동등 문장 2개를 같은 배치에 배치하고 후보 ID 정렬 후 JS divergence 정규화. 의미 검토되지 않은 번역은 자동 활성화하지 않음 |
| 확률 보정 | INT8 배포 logits에 언어×문제 종류×후보 수별 temperature, 표본 부족 시 전역값 사용. 별도 conformal split 유지 |
| Laya 자료 | 공식 연구 코드의 학습 노출 표시를 확인한 AG News, Enron spam, phishing, support tickets, MS MARCO 어댑터. typed-decisions는 teacher-origin으로 분리 |
| 공정 비교 | ND2/Julia/Laya 로컬 실행, 동일 입력 지문, 잘림 거부, 실패 포함 기록, 3모델 공통 지원 집합과 한국어별 paired group bootstrap |
| 데이터 감사 | 같은 상태의 원본 그룹 병합, 공식 test/development와 학습 충돌 시 중단, 토크나이저 길이·선택지 초과 진단 |

기존 C++17 인코더·판단 헤드, 문맥 재사용, proper NLL+Brier+ordinal CRPS,
정확한 decision-training 재시작, pip 인터페이스는 유지합니다.

**Laya multilingual도 mmBERT-base를 사용합니다.** 따라서 같은 인코더 이름이나 더 많은 행만으로
우월성을 주장할 수 없습니다. 한국어 표현·과제 분포를 학습하고 같은 평가에서 검증하는 것이 이번 변경의 목적입니다.

## 2. 설치와 오프라인 확인

Python 3.10+ 및 C++17 컴파일러가 필요합니다. 코드가 있는 폴더에서:

```bash
python -m venv .venv
# Linux/WSL
source .venv/bin/activate
# Windows CMD에서는 대신 .venv\Scripts\activate.bat
python -m pip install .
python -m pip install ".[train,verify,data,test]"
python -m pytest -q
python scripts/smoke_test.py --out runs/smoke
```

PyPI에 게시하지 않았습니다. `pip install .` 또는 첨부한 sdist를 설치하세요.
CUDA 학습용 PyTorch는 장치에 맞게 별도로 설치해야 합니다. CPU 검증 환경은 Python 3.13.5,
PyTorch 2.10.0+cpu입니다. Windows/CUDA 실제 실행은 미검증입니다.
smoke는 바이트 토크나이저를 쓰는 무작위 소형 모델이며 항상 판단을 보류합니다. 그 정확도/속도는 제품 성능이 아닙니다.

## 3. 원본 모델 초기화

```bash
nd2 init --backbone jhu-clsp/mmBERT-base --out init/base
```

`--revision COMMIT_SHA`로 백본을 고정할 수 있습니다. 실제 revision을 기록합니다.
초기화는 원본 Transformers와 독립 인코더의 출력 일치를 검사하고 불일치 시 중단합니다.
검증을 통과하지 않은 가중치에서 조용히 본학습을 진행하지 마세요.

## 4. 공개 데이터 수집·변환

원본을 자동 다운로드하거나, 이미 받은 원본 JSONL/TSV에 `--raw 경로`를 지정할 수 있습니다.
자동 다운로드는 resolved revision/파일 해시를 manifest에 기록합니다.
라이브 다운로드는 이 환경에서 수행하지 않았으므로 원본의 현재 스키마/접근 조건을 확인하세요.

```bash
python scripts/build_data.py --source kornli-snli --out data/kornli-snli.jsonl
python scripts/build_data.py --source kornli-multinli --out data/kornli-mnli.jsonl
python scripts/build_data.py --source klue-nli --out data/klue-nli-train.jsonl
python scripts/build_data.py --source klue-nli --split validation --out data/klue-nli-dev.jsonl
python scripts/build_data.py --source klue-ynat --out data/klue-ynat.jsonl
python scripts/build_data.py --source paws-ko --out data/paws-ko-train.jsonl
python scripts/build_data.py --source paws-ko --split test --out data/paws-ko-test.jsonl
python scripts/build_data.py --source massive-scenario --language ko-KR --out data/massive-ko-train.jsonl
python scripts/build_data.py --source massive-scenario --language ko-KR --split test --out data/massive-ko-test.jsonl
```

MASSIVE는 공식 tar.gz를 직접 읽습니다. HF의 오래된 dataset script 실행에 의존하지 않습니다.
기존에 받은 파일은 `--archive 경로`로 지정하세요. 공식 분할을 유지하며 18개 상황을 모두 후보로 제공합니다.
전체 의도 분류는 `--source massive-intent --labels labels60.json`을 사용합니다.
`labels60.json`은 원본의 60개 의도 문자열을 담은 JSON 배열이어야 합니다.
정답과 19개 오답만 고르는 oracle 후보 축소는 하지 않습니다. 의도 60개 설명은 기본 영어이므로
한국어 설명 비교가 필요하면 원문 기준으로 검토한 설명을 따로 준비해야 합니다.

```bash
# Laya 연구 코드에 학습 노출이 표시된 데이터 원천
python scripts/build_data.py --source ag-news --out data/ag-news.jsonl
python scripts/build_data.py --source enron-spam --out data/enron.jsonl
python scripts/build_data.py --source phishing --out data/phishing.jsonl
python scripts/build_data.py --source support-queue --out data/support.jsonl
python scripts/build_data.py --source msmarco --out data/msmarco.jsonl

# typed-decisions: 교사 확률 기반 합성 자료. 인간 정답 자료와 별도 실험!
python scripts/build_data.py --source typed-decisions --split test --revision c76749ec58bd8c3d2ea706b31c333a9059c38f90 --out data/typed-test.jsonl
```

전체 원천 목록과 학습/평가 역할은 `docs/LAYA_DATA_AND_KOREAN.md`를 읽으세요.
MS MARCO의 `is_selected=0`은 모든 가능한 관련성에 대한 확정 음성이 아니므로,
어댑터는 '근거로 선택되었는가'라는 약한 selection label로 기록합니다.

### 같은 원본·번역·중복을 올바르게 묶기

`group_id`는 동일 원본 문서/대화/병렬 번역 전체가 공유해야 합니다.
PAWS-X와 MASSIVE의 원본 ID는 언어 간 공유합니다. KorNLI와 영어 SNLI/MNLI를 함께 쓰려면
원본 ID 대응을 별도로 검증해야 합니다. 번역문의 텍스트 해시만으로 영어 원본을 찾아낼 수 없습니다.
검증되지 않은 영문·번역 자료를 서로 다른 split에 놓지 마세요.

```bash
python scripts/reconcile_groups.py --input data/klue-nli-train.jsonl data/klue-nli-dev.jsonl data/paws-ko-train.jsonl data/paws-ko-test.jsonl data/massive-ko-train.jsonl data/massive-ko-test.jsonl --out data/combined.jsonl --repartition-training
```

이 명령은 SQLite에서 동일 NFC 상태를 가진 그룹을 합칩니다. 원본 train 내부의 생성 split만
재배정할 수 있습니다. 공식 test/development와 충돌하면 중단합니다. 이때 test를 옮기는 대신
어느 학습 원본을 제외할지 감사해야 합니다. 근접 중복 탐지/번역 정렬을 완성한 도구는 아닙니다.
기존 `nd2 prepare`도 충돌 시 거부합니다. `reconcile_groups`는 파일을 합치는 선택 도구입니다.

## 5. 한국어 인코더 추가학습 — 선택 사항

본학습 전 별도 한국어 원문·영어 유지 학습 원문으로 masked LM을 실행할 수 있습니다.
언어를 번역하여 읽는 추론 단계가 아니라, 인코더 자체의 가중치를 추가로 학습합니다.
원본 문서별로 train/validation을 나누고 최종 벤치마크/보정 문서는 넣지 마세요.

`corpus.jsonl` 형식:

```json
{"text":"사용 권한이 있는 실제 한국어 문서의 본문", "group_id":"document-001", "partition":"train", "language":"ko"}
{"text":"A different authorized English document.", "group_id":"document-002", "partition":"validation", "language":"en"}
```

위 두 줄은 형식 예시이지 학습 코퍼스가 아닙니다. validation에도 각 목표 언어가 충분히 있어야 합니다.
초기 구성안으로 한국어 원문과 영어 유지 자료를 혼합하되, 비중은 고정된 실측 최적값이 아닙니다.

```bash
python scripts/adapt_encoder.py prepare --input corpus.jsonl --init init/base --out data/corpus.sqlite --block-length 256
python scripts/adapt_encoder.py train --init init/base --corpus data/corpus.sqlite --out runs/ko-adapt --device cuda --bf16 --gradient-checkpointing --batch-size 2 --mask-chunk 32
```

완료 후 `runs/ko-adapt/best`를 아래 `--init`에 사용합니다. 추가학습을 생략하면 `init/base`입니다.
마스킹 비율은 15%, 80/10/10 교체, 기존 subword 토큰 단위입니다. 형태소 분석/whole-word masking 구현이라고 주장하지 않습니다.
MLM 출력은 입력 임베딩에 묶고 마스킹 위치만 chunk 단위로 계산합니다. 대형 어휘의 연산 비용은 남습니다.
추가학습용 transform은 배포하지 않아 결정 모델 파라미터 수는 그대로입니다.
현재 이 추가학습 스크립트에는 정확한 중간 재개가 없습니다. decision-training의 `--resume`과 구분하세요.
추가학습이 성능을 떨어뜨릴 수도 있으므로, 같은 판단 데이터로 '추가학습 없음' 대조군을 반드시 유지합니다.

## 6. 한국어 중심 판단 학습

```bash
nd2 prepare --input data/combined.jsonl --init init/base --out data/cache --max-state 2048 --max-pair 256
python scripts/audit_tokenizer.py --init init/base --input data/combined.jsonl --out reports/tokenizer.json
nd2 train --init init/base --cache data/cache --out runs/ko-base --device cuda --bf16 --gradient-checkpointing --epochs 3 --batch-size 4 --accumulation 16 --language-weights '{"ko":2.0,"en":1.0}' --selection-metric macro_language_nll
```

셸 예시는 bash/WSL입니다. Windows CMD에서는 JSON 따옴표를 셸에 맞게 전달하세요.
`ko:2`는 한국어 예제의 loss weight를 두 배로 하는 것이지 데이터의 2/3가 한국어가 된다는 뜻이 아닙니다.
`training_report.json`에 실제 언어별 행 노출과 유효 가중치를 기록합니다.
`macro_language_nll`은 언어별 평균 NLL을 같은 비중으로 평균합니다. 특정 과제만 반복해 점수를
높이지 않도록 언어별/과제별 평가와 원본 그룹을 별도로 확인해야 합니다.

### 검토된 한·영 대응 예제의 일관성 학습

정말 같은 판단 문제임을 검토한 두 행에만 다음을 지정합니다.

```json
{"equivalence_id":"reviewed-pair-001", "equivalence_verified":true, "group_id":"same-original-001", "language":"ko"}
```

다른 행은 같은 equivalence_id/group_id와 `language:"en"`을 사용합니다.
후보 ID·정답 분포·점수 레벨이 같아야 하며, 두 언어 행이 정확히 2개여야 합니다.
전체 state/question/criteria도 의미적으로 대응하는지 사람이 검토해야 합니다.
이 metadata는 기존 JSONL 행에 더하는 필드이며 위 줄만으로 완전한 학습 행이 되는 것은 아닙니다.
MASSIVE 변환기는 대응 ID는 넣지만 `equivalence_verified:false`로 두므로 자동 승인되지 않습니다.

검토된 쌍이 있을 때 학습 명령에 `--pair-consistency 0.05`를 추가합니다.
후보 셔플 후에도 ID로 확률을 정렬해 JS divergence를 구합니다. 검토된 쌍이 없으면 오류로 중단합니다.
이 정규화는 선택 사항이며 실제로 동등하지 않은 두 문제를 묶으면 학습을 해칠 수 있습니다.

## 7. 배포와 언어별 확률 보정

```bash
nd2 export --checkpoint runs/ko-base/best --out bundles/ko-int8
nd2 evaluate --bundle bundles/ko-int8 --cache data/cache --split temperature --out reports/temperature.jsonl
nd2 evaluate --bundle bundles/ko-int8 --cache data/cache --split conformal --out reports/conformal.jsonl
nd2 calibrate --bundle bundles/ko-int8 --temperature reports/temperature.jsonl --conformal reports/conformal.jsonl --language-calibration --min-bucket-groups 200 --alpha 0.1
nd2 evaluate --bundle bundles/ko-int8 --cache data/cache --split test --out reports/test.jsonl
```

언어×Choice/Noul/Score×후보 수 범위로 온도를 맞추고, 독립 원본 200그룹 미만인 셀은 전역값을 씁니다.
실제 INT8 배포 모델의 logits로 fit하며, test에서는 절대 맞추지 않습니다.
Conformal은 고정된 온도를 적용한 별도 split의 그룹 최대 오차로 계산합니다.
**언어별 보정을 했다고 한국어 개별 답이나 수락된 답의 90% 정확도가 보장되는 것은 아닙니다.**

```python
from nanodecision2 import DecisionEngine
engine = DecisionEngine("bundles/ko-int8")
answer = engine.decide(
    state="이전 테스트는 통과했지만 이번 변경 후에는 테스트하지 않았다.",
    questions={"done": {"type":"noul", "question":"이번 변경의 검증이 완료되었는가?",
                         "criteria":{"false":"아직 검증하지 않았다.", "true":"변경 후 검증을 완료했다."}}},
    language="ko",
)["answers"]["done"]
print(answer["probabilities"], answer["abstained"])
```

언어를 지정하지 않으면 전역 보정을 사용합니다. 확률만 보고 언어를 추정하지 않습니다.
토크나이저 어휘나 인코더 구조를 바꾸지 않아 native 런타임을 그대로 사용합니다.
INT8은 가중치 저장이며 계산은 FP32입니다. native CPU 외 GPU inference를 추가한 버전은 아닙니다.

## 8. Julia + Laya + 우리 모델 비교

**평가 코드와 합성 fixture는 검증했습니다. 실제 Julia/Laya 가중치 비교 결과는 아직 없습니다.**
각 패키지의 의존성이 다르므로 별도 가상환경에서 실행하세요. 세 환경에 동일한 이 소스도 설치하면
같은 `evaluate_raw.py`를 사용할 수 있습니다. 외부 SDK와 가중치는 각각 공식 저장소에서 받습니다.
Laya 코드 검토 기준 commit은 `9d955671415fc19f069b9cc998928075c1f255ec`입니다.

```bash
# 세 모델에 정확히 같은 JSONL 전달; 한국어 기본 대조군은 Laya multilingual
python scripts/evaluate_raw.py --engine nd2 --model bundles/ko-int8 --input data/massive-ko-test.jsonl --out reports/nd2-ko.jsonl --device cpu
python scripts/evaluate_raw.py --engine julia --model /path/to/Julia-1 --input data/massive-ko-test.jsonl --out reports/julia-ko.jsonl --device cpu --max-length 2048 --head-length 512
python scripts/evaluate_raw.py --engine laya --laya-variant multilingual --model /path/to/laya-multilingual --input data/massive-ko-test.jsonl --out reports/laya-ko.jsonl --device cpu --max-length 2048 --head-length 512
python scripts/compare_predictions.py --predictions reports/nd2-ko.jsonl reports/julia-ko.jsonl reports/laya-ko.jsonl --out reports/three-model-ko.json
```

위 경로는 실제 다운로드 위치로 바꿔야 합니다. benchmark 결과의 모델 파일 해시·라이브러리 버전·입력 SHA를 보관합니다.
가중치는 이 ZIP에 없습니다. `/path/to/...`를 자동으로 찾아 주거나 API 호출로 대체하지 않습니다.

- 한국어 18상황은 세 모델이 전체 후보를 받는 공통 트랙입니다. 기본 비교는 forced-choice 확률/argmax이며,
  각 제품의 act/escalate 정책이 서로 같다고 가정하지 않습니다.
- 60의도/77후보 전체 비교는 별도 용량 트랙입니다. Julia의 20후보 한계는 오류로 남기며 몰래 후보를 줄이지 않습니다.
- Laya의 자체 build_sequence 결과가 질문·후보·상태의 완전한 표현과 같아야 합니다. 설명 잘림/후보 붕괴는 실패로 기록합니다.
- 품질 차이는 세 모델 모두 지원한 같은 행의 교집합에서만 계산합니다. 전체 요청의 성공/실패/지원 비율도 함께 보고합니다.
- 언어별 결과와 같은 원본 그룹을 재표집한 bootstrap 구간을 제공합니다. 여러 지표를 골라 유리한 것만 승리라고 하지 마세요.
- ND2 INT8 CPU와 baseline FP32 CPU의 배포 비교, ND2 FP32 CPU와 baseline FP32 CPU의 정밀도 대조는 별도 보고합니다.
- 시간은 준비/토큰화/계산/결과변환을 포함하고 load/warmup/파일 IO는 제외합니다. 전체 프로세스 peak RSS는 load와 기록 메모리도 포함합니다.
- 공개 typed specialist 성적은 범용 미학습 과제 성적과 별개입니다. `reference_option_id`와 teacher probability mode를 구분합니다.

`examples/ko_contrast_checks.jsonl`에는 부정·조사·시간·예외 등 32개 합성 개발 점검 문항이 있습니다.
이를 블라인드/실사용 정답 데이터로 취급하지 마세요. `docs/BENCHMARK_PROTOCOL.md`에 상세 비교 설계가 있습니다.

## 9. 검증과 제한

`reports_v030/VERIFIED.md`와 테스트 로그를 확인하세요. 큰 인코더를 실제로 한국어 추가학습한 것,
100만 행 정제 완료, Julia/Laya 초과 성능, 대형 모델 CPU RSS/처리량, CUDA/Windows는 미검증입니다.
기존 `reports/`는 0.2.0의 역사적 기록이며 새 버전 실측으로 바꾸어 쓰지 않습니다.

본 소스는 MIT, 외부 코드/모델/데이터는 각 원천의 이용 조건을 따릅니다. 민감한 원문과 개인 식별정보는 수집 단계에서 관리하세요.
