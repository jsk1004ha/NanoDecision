# NanoDecision2 0.2.0

Jev API 없이 학습하는 구조화된 확률 판단 모델과 C++17 CPU 실행 엔진.

**이 배포물은 학습·검증·배포 코드입니다. Julia 1을 능가한다고 검증된 학습 가중치가 아닙니다.**
기본 백본은 공개된 `jhu-clsp/mmBERT-base`이며 초기화 시 다운로드합니다. 의사결정 헤드는 새로 학습해야 합니다. 네트워크가 차단된 제작 환경에서는 대형 사전학습 가중치·실제 데이터 다운로드·CUDA 학습·Julia 비교를 수행하지 못했습니다. `--smoke`는 작은 무작위 모델을 사용하는 통합 테스트 전용입니다.

## 1. 구현한 내용

- **학습:** PyTorch, 사전학습 인코더 전체 미세조정, 인코더/헤드별 학습률, 층별 학습률 감쇠, AdamW, warmup/cosine, BF16 선택, gradient checkpointing, 길이별 배치, 실제 배치 가중치로 정규화하는 gradient accumulation, validation NLL 기반 선택, 옵티마이저/RNG까지 저장하는 재시작.
- **확률:** 로그 손실 + Brier + 순서형 CDF 손실. 실제 양자화 모델의 별도 temperature scaling. 독립적인 conformal 데이터에서 원본 그룹별 최대 nonconformity를 사용하는 prediction set. 기대 비용 최소화 API.
- **인코더:** ModernBERT 수학과 대응하는 mmBERT 가중치 가져오기. 공통 상태의 토큰 표현을 유지하며 질문+후보별 표현이 두 차례 근거를 참조합니다. 후보 집합의 평균/최대 표현을 반영하는 permutation-equivariant 판단 헤드.
- **데이터:** JSONL 스트리밍 전처리, SQLite 중복·그룹 감사, mmap 토큰 캐시/오프셋 인덱스, 분할 지문과 입력 파일 해시, 원본 그룹 단위 누출 검사. 숫자만 늘리는 합성 데이터 생성기는 본학습용으로 제공하지 않습니다.
- **실행:** 독립 C++ 인코더 + 판단 헤드. CPython 확장과 pip 패키지. 텍스트 토큰화는 Rust 기반 Hugging Face `tokenizers`. 실행 경로에 torch/NumPy/Transformers/ONNX Runtime을 import하지 않습니다.
- **평가:** 동일 정답/후보를 쓰는 Julia 로컬 비교 어댑터, 원본 그룹 단위 paired bootstrap, NLL/Brier/ECE/순서형 오차, 고정 confidence 임계값과 conformal singleton의 관측 정확도/처리 비율.

## 2. 설치

Python 3.10 이상과 C++17 컴파일러, 해당 Python 개발 헤더가 필요합니다. 실제 검증은 Linux x86-64, Python 3.13.5, PyTorch 2.10.0+cpu에서 수행했습니다. Windows/MSVC와 CUDA 경로는 구현했지만 이 환경에서 실행 검증하지 않았습니다.

압축을 풀어 **pyproject.toml이 있는 프로젝트 디렉터리**에서 실행합니다.

```bash
python -m venv .venv
# Linux/WSL:
source .venv/bin/activate
# Windows CMD에서는 위 줄 대신: .venv\Scripts\activate.bat

# 실행만 설치: C++ 확장 + Rust 토크나이저 바인딩
python -m pip install .

# 학습·공개 데이터·사전학습 대응 검증·테스트까지 설치
python -m pip install ".[train,verify,data,test]"
python -m pytest -q
```

PyPI에 게시한 패키지가 아닙니다. `pip install nanodecision2`라는 공개 이름으로 내려받는 방법이 아니라 **로컬 소스나 첨부한 sdist를 설치**하는 방법입니다. CUDA를 쓰려면 먼저 자신의 드라이버/장치에 맞는 공식 PyTorch CUDA 휠을 설치하세요. 위 extras가 CUDA 사용 가능성을 보장하지 않습니다.

기본 네이티브 빌드는 외부 수학 라이브러리가 없는 이식성 위주의 스칼라 참조 커널입니다. **가벼운 의존성이지, 최고 처리 속도를 보장하는 구현은 아닙니다.** CBLAS 헤더와 BLAS 라이브러리가 있는 Linux/WSL에서는 다음으로 행렬 연산을 연결할 수 있습니다.

```bash
ND2_BLAS=1 python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

예: Linux에 `libblas-dev` 또는 호환 개발 패키지 필요. OpenBLAS 등 실제 연결된 BLAS와 CPU에 따라 속도가 달라집니다. 전체 모델 속도와 RSS를 실측한 뒤 배포하세요. 별도 가상환경을 쓰면 학습용 대형 라이브러리를 배포 환경에서 제외할 수 있습니다.

## 3. 오프라인 통합 확인

```bash
python scripts/smoke_test.py --out runs/smoke
```

104개의 짧은 합성 **파이프라인 확인** 행을 만들고, CPU 학습 → INT8 저장 → temperature fitting → conformal fitting → 테스트 → C++ 추론을 실행합니다. `runs/smoke/report.json`에 결과가 남습니다. 이 데이터나 모델로 언어 능력/실제 업무 성능을 판단하면 안 됩니다. smoke 모델은 항상 `abstained=true`를 반환합니다.

## 4. 실제 사전학습 모델 초기화

```bash
nd2 init --backbone jhu-clsp/mmBERT-base --out init/base
```

Jev 키가 필요하지 않습니다. 공개 백본과 토크나이저만 다운로드합니다. `--revision`에 Hugging Face commit SHA를 넣어 재현성을 고정할 수 있으며 실제 resolved revision도 저장합니다. `trust_remote_code`를 사용하지 않습니다.

초기화는 가중치를 독립 구현에 대응시킨 뒤, 원본 Transformers 인코더와 CPU FP32 출력을 비교합니다. 불일치하면 중단합니다. 제작 환경에서는 실제 백본 다운로드가 불가능하여 이 검증 자체는 실행하지 못했습니다. `--skip-upstream-check`는 명시적으로 검증을 건너뛰는 연구용 옵션이며 권장 기본 경로가 아닙니다.

용량 비교 실험에는 다음도 지원합니다.

```bash
nd2 init --backbone jhu-clsp/mmBERT-small --out init/small
```

base는 성능 우선 출발점, small은 자원-성능 비교군입니다. 새 가중치에 대한 정확도가 검증되기 전에는 어느 것이 실제로 더 낫다고 단정하지 않습니다.

## 5. 실제 학습 데이터

JSONL 한 줄에 하나의 판단 문제를 작성합니다. 동일 문서/세션, 번역, 바꿔 쓰기, 여러 질문은 같은 `group_id`를 공유해야 합니다.

```json
{"id":"case-001-done","group_id":"session-001","state":"코드 수정은 끝났지만 변경 이후 테스트는 아직 실행하지 않았다.","type":"noul","question":"현재 변경을 검증 완료로 판단할 수 있는가?","criteria":{"false":"새 변경의 검증이 끝나지 않았다.","true":"새 변경의 검증이 성공적으로 끝났다."},"target":false}
```

Choice는 `criteria`에 ID:설명 매핑, `target`에 해당 ID를 사용합니다. Noul은 `target`에 JSON boolean을 사용합니다. Score는 `levels`에 `{value, description}`을 작은 값부터 나열하고 `target`은 그 레벨 중 정확히 하나여야 합니다. 임의의 연속 점수를 가짜 확률분포로 보간하지 않습니다. 분포 정답은 `target_probs`를 명시합니다. 이는 실제 라벨러 분포/검증된 분포라는 출처가 필요하며 LLM의 “확신 90%”라는 문장을 그대로 옮기면 안 됩니다.

`id`와 선택지 ID는 모델 입력 문장에 넣지 않습니다. 긴 입력은 오류로 보고하며 몰래 자르지 않습니다.

### 공개 데이터 변환

```bash
python scripts/fetch_public.py --source snli --out data/snli.jsonl
python scripts/fetch_public.py --source multinli --out data/multinli.jsonl
python scripts/fetch_public.py --source klue-nli --out data/klue-nli.jsonl
# xLAM은 현재 Hugging Face 이용 조건 수락/로그인이 필요할 수 있습니다.
python scripts/fetch_public.py --source xlam --out data/xlam-single-tool.jsonl
```

원본 train 분할만 읽고 실제 쓴 행 수/건너뛴 행 수를 manifest로 남깁니다. NLI는 entailment/neutral/contradiction의 의미를 보존합니다. xLAM은 단일 도구 정답만 변환하며 다중 호출을 하나의 잘못된 정답으로 만들지 않습니다. 이 어댑터들의 라이브 다운로드는 제작 환경의 DNS 제한으로 미검증입니다.

**이 네 자료를 합친 것만으로 범용 의사결정용 100만~300만 행이 완성되지는 않습니다.** 완료 판단, 평가 기준에 따른 점수, 대화/로그 상태, 긴 근거, 정보 부족 사례 등 실제 목적의 자료를 추가하세요. 동봉된 것은 수집/변환 코드이며 100만 행의 정제된 데이터나 학습된 가중치가 아닙니다.

### 캐시와 분할

```bash
nd2 prepare --input data/snli.jsonl data/multinli.jsonl data/klue-nli.jsonl data/my-decisions.jsonl --init init/base --out data/cache --max-state 2048 --max-pair 256
```

`my-decisions.jsonl`은 사용 목적의 실제 자료이며 직접 제공해야 합니다. 해당 파일을 아직 만들지 않았다면 존재하는 파일만 명령에 넣으세요. `partition`을 명시하지 않은 원본 그룹은 약 train/validation/temperature/conformal/test = **70/10/5/5/10**으로 나뉩니다. 총 원본 100만 행과 학습 100만 행은 다릅니다. 실제 수량은 `data/cache/manifest.json`에서 확인합니다.

`partition`을 명시하면 정해 둔 공식 분할을 유지할 수 있습니다. 같은 그룹을 서로 다른 분할에 넣으면 오류입니다. 근접 중복·번역·문서 관계를 의미적으로 자동 탐지하는 기능은 없으므로 올바른 그룹 정보는 데이터 작성자의 책임입니다. 그룹 간 일반화 테스트를 별도로 구성하세요.

학습은 토큰화된 전체 문장을 Python 객체로 상주시켜 두지 않습니다. JSONL과 인덱스는 mmap으로 읽습니다. 파일을 mmap했다고 실제 사용한 모든 페이지가 RSS에 잡히지 않는다는 뜻은 아닙니다. 길이·셔플 인덱스와 현재 배치는 추가 메모리를 사용합니다.

## 6. 학습

```bash
nd2 train --init init/base --cache data/cache --out runs/base --device cuda --bf16 --gradient-checkpointing --epochs 3 --batch-size 4 --accumulation 16 --encoder-lr 0.00002 --head-lr 0.0001 --layer-decay 0.9 --brier-weight 0.2 --rps-weight 0.2
```

위 값은 실측 최적값이 아닌 시작 설정입니다. 배치의 선택지 수, 문맥 길이, GPU에 따라 메모리가 달라집니다. OOM이면 micro-batch부터 줄이세요. CPU에서는 `--device cpu`를 사용하고 `--bf16`을 빼세요.

정확도와 NLL이 validation에서 개선되는지 먼저 확인하고 10만 → 100만 → 300만 학습 문제로 확장하는 실험을 권합니다. 대규모 데이터에 예전 864행 데모의 수초 학습시간을 외삽하지 마세요. 현재 코드의 학습 보고 시간은 모델/캐시 준비 이후 구간입니다.

재시작 시 같은 데이터·설정으로 다음을 사용합니다.

```bash
nd2 train --init init/base --cache data/cache --out runs/base --device cuda --bf16 --gradient-checkpointing --epochs 3 --batch-size 4 --accumulation 16 --resume runs/base/last.pt
```

기본값에서 바꾼 다른 인자도 원래 값 그대로 전달하세요. 데이터나 옵티마이저 설정이 바뀌면 exact resume이 거부됩니다. `last.pt`는 신뢰하는 자기 학습 체크포인트만 읽으세요. 모델 배포용 저장은 safetensors/전용 숫자 형식입니다. CPU에서는 중단 후 재개와 연속 학습의 가중치가 비트 단위로 동일한지 테스트했습니다. CUDA의 비결정적 연산까지 동일성을 보증하지 않습니다.

### 문맥 길이 확대

8K는 기본 백본의 최대 위치 길이이며 이 배포 모델의 8K 정확도 검증 결과가 아닙니다. 초기 캐시가 2048이면 배포 제한도 2048로 남습니다. 설정만 늘려 8K 검증 완료라고 하지 않습니다.

동일 원본 파일·동일 그룹 분할에서 8192 캐시를 다시 만든 뒤, 기존 best를 `--init`으로 쓰는 **추가 학습**은 가능합니다. 이는 옵티마이저 exact resume이 아니라 warm start입니다. 입력 자료나 그룹 배정이 바뀌면 교차 데이터 누출을 막기 위해 거부합니다. 처음 캐시에 2048 초과 행이 있으면 prepare 자체가 오류를 내므로, 그런 자료는 처음부터 적절한 길이 예산으로 준비해야 합니다.

## 7. 배포, 확률 보정, 평가

```bash
nd2 export --checkpoint runs/base/best --out bundles/base-int8
nd2 evaluate --bundle bundles/base-int8 --cache data/cache --split temperature --out reports/temperature.jsonl
nd2 evaluate --bundle bundles/base-int8 --cache data/cache --split conformal --out reports/conformal.jsonl
nd2 calibrate --bundle bundles/base-int8 --temperature reports/temperature.jsonl --conformal reports/conformal.jsonl --alpha 0.1
nd2 evaluate --bundle bundles/base-int8 --cache data/cache --split test --out reports/test.jsonl
nd2 predict --bundle bundles/base-int8 --input examples/request.json
```

FP32 대조군은 `nd2 export ... --fp32`로 별도 번들에 내보냅니다. **양자화 뒤 실제 배포 모델에서 확률을 다시 맞춥니다.** calibration은 모델 파일 SHA-256에 연결되며 파일이 바뀌면 사용을 거부합니다.

`alpha=0.1`은 완전한 원본 그룹이 교환 가능하다는 조건에서 새 그룹의 정답들을 prediction set이 동시에 포함하는 주변적 보장에 관련된 값입니다. “허용된 개별 답은 90% 이상 정확하다”는 뜻이 아닙니다. 상세 내용은 `docs/THEORY.md`를 읽으세요. singleton이 아닌 집합은 기본적으로 판단을 보류합니다. 보류가 악성 입력이나 분포 이동을 완벽히 감지하지는 않습니다.

## 8. Python에서 사용

```python
from nanodecision2 import DecisionEngine

engine = DecisionEngine("bundles/base-int8")
result = engine.decide(
    state="코드 수정은 끝났지만 변경 이후 테스트는 아직 실행하지 않았다.",
    questions={
        "done": {
            "type": "noul",
            "question": "변경 사항의 검증이 완료되었는가?",
            "criteria": {
                "false": "변경 후 검증이 아직 끝나지 않았다.",
                "true": "변경 후 검증을 수행했고 모두 통과했다.",
            },
        }
    },
)
answer = result["answers"]["done"]
print(answer["probabilities"], answer["prediction_set"])
if answer["abstained"]:
    print("판단 보류", answer["abstain_reasons"])
else:
    print(answer["value"])
```

실제 출력 값은 학습 결과에 따릅니다. 예제에 고정된 정답이나 신뢰도를 하드코딩하지 않았습니다. `noul`에는 참일 확률, `score.expected_value`에는 점수 기댓값이 들어갑니다. 보류 상태의 `suggested_value`/`noul`/`expected_value`를 자동 실행에 그대로 쓰지 마세요.

한 상태를 여러 번 질의하려면 `prepared=engine.prepare(state)` 후 `engine.score_prepared(prepared, question)`을 사용합니다. `decide`는 1~64개 질문을 받아 상태를 한 번 인코딩합니다. 질문별 후보는 2~256개입니다. **이는 입력/수학 구현의 제한이며, 그 규모의 정확도나 처리량 보장이 아닙니다.** 후보 자체는 처음에는 각각 인코딩하며 반복되는 후보 표현은 한도 내 캐시합니다. 질문 추가가 다른 질문의 logits를 바꾸지 않도록 분리되어 있습니다.

프로세스를 유지하는 JSONL 표준 입출력:

```bash
nd2 stdin --bundle bundles/base-int8
```

## 9. 순수 C++ 실행

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
# BLAS 빌드: 위 configure 명령에 -DND2_BLAS=ON 추가
```

`nd2-native`는 Python 없는 실행 파일입니다. 입력은 **이미 토큰화한 정수 ID**를 사용하는 줄 단위 프로토콜입니다. 텍스트/JSON API는 Python + Rust tokenizer 래퍼를 사용합니다. C++ CLI 안에 Gemma 토크나이저를 다시 구현했다고 주장하지 않습니다.

```text
n_state state_ids... n_options n_option_1 option_1_ids... n_option_2 option_2_ids... ...
```

```bash
native/build/nd2-native bundles/base-int8/model.nd2 1.0 < requests.tokens
```

두 번째 인자는 temperature입니다. CLI는 raw logits/softmax만 반환합니다. 올바른 보정 temperature와 conformal/비용 정책을 직접 연결해야 하며, 완성된 정책 API가 필요하면 Python 래퍼를 쓰세요. 모델 파일은 로드 중 수정하면 안 됩니다.

## 10. 메모리와 크기

실제 코드의 shape에서 계산한 파라미터 수:

| 설정 | 파라미터 | 행렬 INT8 + 행별 scale + 기타 FP32의 이론 payload |
|---|---:|---:|
| base + head | 308,252,673 | 309,934,344 bytes |
| small + head | 141,511,809 | 142,972,680 bytes |

위는 **계산값**이며 실측 전체 모델 파일/RSS가 아닙니다. 파일 정렬·메타데이터, 토크나이저, 활성값, 캐시, Python 또는 C++ allocator가 별도로 필요합니다.

큰 임베딩과 선형 가중치는 메모리에서도 INT8 저장을 유지합니다. 임베딩은 사용한 행만 FP32로 복원하고 BLAS 모드에서는 선형 가중치를 작은 타일 단위로 복원합니다. 계산은 FP32입니다. 정수 전용 INT8 GEMM이나 4배 가속이라고 표현하지 않습니다. 전역 attention도 연산량은 여전히 길이 제곱에 비례하지만 전체 attention 확률 행렬을 메모리에 쌓지 않고 행별로 계산합니다.

C++ 자체가 모델 용량을 줄이는 것은 아닙니다. 큰 가중치의 정밀도, 중복 복사, 작업 버퍼, 실행 라이브러리를 함께 바꾼 점이 중요합니다. 네이티브 구현을 공개 서비스에 쓰기 전에는 실제 모델/입력으로 속도·RSS·동시성·신뢰하지 않는 입력에 대한 추가 검증이 필요합니다.

## 11. Julia와 비교

`docs/BENCHMARK_PROTOCOL.md` 참조. `scripts/evaluate_raw.py`에 로컬 Julia 어댑터가 있습니다. Jev API는 사용하지 않습니다. Julia 가중치/공식 패키지를 별도로 받아 설치해야 합니다.

```bash
python scripts/evaluate_raw.py --engine nd2 --model bundles/base-int8 --input data/heldout.jsonl --out reports/nd2-heldout.jsonl
python scripts/evaluate_raw.py --engine julia --model /path/to/Julia-1 --input data/heldout.jsonl --out reports/julia-heldout.jsonl
python scripts/compare_predictions.py --a reports/nd2-heldout.jsonl --b reports/julia-heldout.jsonl --out reports/paired.json
```

동일 입력/criteria/후보/정답, 원본 그룹 단위 paired bootstrap을 사용합니다. 공개 typed benchmark의 Score 정확도는 기대 점수 반올림이 아니라 최고 확률 옵션으로 계산합니다. Julia의 native 후보 제한을 넘으면 어댑터는 오류를 내며 정답 후보를 몰래 남기거나 불공정한 후보 축소를 하지 않습니다. 이 비교 어댑터는 실제 Julia 가중치로 실행하지 못했으므로 라이브 검증이 필요합니다.

## 12. 검증 결과와 한계

`reports/VERIFIED.md`, `reports/native_verification.json`, `reports/pytest.txt`를 확인하세요. 작은 모델로 학습/재개/네이티브 일치/독립 C++/FP32 및 INT8 경로를 실행했습니다. **대형 사전학습 모델을 실제로 가져오는 전체 경로, 100만 행 본학습, Julia 우월성, Windows, CUDA, 8K 과제 정확도는 검증하지 않았습니다.** 테스트 통과는 모델 능력의 증거가 아닙니다.

모델과 tokenizer는 원래 배포 조건을 따릅니다. 본 구현 소스는 MIT입니다. 참고 자료와 논문은 `docs/REFERENCES.md`에 있습니다.
