# NanoDecision

**텍스트·문서·실행 상태를 읽고, 선택·참일 확률·점수로 답하는 로컬 판단 모델.**

NanoDecision은 문장을 길게 생성하는 대신, 입력 상태와 사용자가 정의한 질문·후보를 받아 구조화된 판단과 확률분포를 반환합니다. **Auto5M 0.3.1**은 공개 데이터 다운로드부터 500만 판단 문제 구성, 학습, 확률 보정, C++ 실행 모델 저장까지 하나의 실행 흐름으로 연결합니다.

**저장소:** `jsk1004ha/NanoDecision` · **Python 패키지:** `nanodecision2` · **CLI:** `nd2` · **버전:** `0.3.1`

[빠른 시작](#빠른-시작) · [데이터 구성](#데이터-구성) · [모델 사용](#모델-사용) · [검증 범위](#검증-범위) · [상세 설명서](README_ko.md)

> 현재 제공물은 학습·추론 **소스 코드**입니다. 대규모 학습을 마친 가중치는 포함하지 않습니다. 소형 모델의 전체 파이프라인과 테스트는 실행했지만, 실제 500만 행 확보·CUDA 본학습·Julia/Laya 대비 우월성은 아직 검증하지 않았습니다.

## 주요 기능

| 영역 | 구현 |
|---|---|
| 타입이 정해진 판단 | `choice`는 후보별 확률, `noul`은 참일 확률, `score`는 순서가 있는 점수 분포와 기댓값을 반환합니다. |
| 사전학습 인코더 | 기본 백본은 `jhu-clsp/mmBERT-base`입니다. 질문과 후보가 토큰 단위 문맥 표현을 참조합니다. |
| 다중 질문 | 한 상태의 문맥 표현을 재사용합니다. 서로 다른 질문의 답은 별도로 계산합니다. |
| 확률 학습 | NLL·Brier 손실과 점수형 문제의 누적분포 손실을 사용합니다. |
| 영어·한국어 평가 | 영어 기준 모델을 학습한 뒤, 영어와 한국어 검증 성능의 회귀를 검사하며 혼합학습 후보를 선택합니다. |
| 대규모 데이터 준비 | 원본 그룹·분할·라벨·중복·입력 길이를 검사하고 SQLite와 압축 토큰 캐시를 사용합니다. |
| 재개 | 원천 revision, 파일 해시, 캐시 및 학습 체크포인트를 기록합니다. 동일 명령으로 완료 단계를 건너뛰고 이어갑니다. |
| 네이티브 배포 | C++17 신경망 엔진, Rust 기반 토크나이저 바인딩, Python API를 제공합니다. 추론 경로에는 PyTorch가 필요하지 않습니다. |

Jev API나 유료 교사 API는 필요하지 않습니다. 공개 사전학습 인코더와 공개 판단 자료를 활용하는 독립 구현이며, Jev의 내부 구조나 RLCD를 재현했다고 주장하지 않습니다.

## 빠른 시작

### 1. 환경 준비

자동 본학습은 **Linux 또는 WSL Ubuntu**, **Python 3.11~3.13**, **C++17 컴파일러**, **CUDA/BF16 지원 GPU**를 요구합니다. GPU가 없으면 CPU 본학습으로 조용히 전환하지 않습니다.

Ubuntu에 개발 도구가 없다면 처음 한 번 설치합니다.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential libopenblas-dev
```

NVIDIA 드라이버와 WSL 자체는 별도로 준비해야 합니다. 실행기가 운영체제나 드라이버를 변경하지 않습니다. 기본 실행 전 작업 디스크의 **150GiB 이상 여유 공간**을 검사합니다. 이는 실측 사용량이 아니라 다운로드·중간 데이터·여러 체크포인트를 위한 시작 예산입니다.

### 2. 소스 받기

```bash
git clone https://github.com/jsk1004ha/NanoDecision.git
cd NanoDecision
```

### 3. 다운로드부터 학습까지 실행

```bash
bash run_all.sh --workdir "$HOME/nd2-runs/auto5m"
```

실행기는 `.venv031`을 생성하고, 고정 의존성·CUDA용 PyTorch·C++ 확장을 설치한 뒤 학습 파이프라인을 시작합니다. 이미 동일한 소스와 환경으로 설치를 마쳤으면 설치 단계를 재사용합니다. 작업 폴더는 WSL의 Linux 파일시스템 내부에 두는 구성을 권장합니다.

```text
환경·GPU·디스크 검사
    → mmBERT-base 및 토크나이저 다운로드·원본 출력 대조
    → 공개 원천 다운로드·변환·정제
    → 학습 500만 행과 별도 평가 분할 구성
    → 영어 기준 모델 학습
    → 한·영·다국어 혼합학습
    → 내보내기·확률 보정·언어별 채택 검사
    → 최종 평가·bundle 저장
```

데이터 구성까지만 먼저 실행할 수도 있습니다.

```bash
bash run_all.sh --workdir "$HOME/nd2-runs/auto5m" --stop-after-stage data
```

이후 같은 명령에서 `--stop-after-stage data`를 빼면 학습부터 이어집니다.

### 작은 통합시험부터 실행

대규모 다운로드와 GPU 학습 없이 코드 경로를 확인하는 별도 CPU 환경입니다. **이 명령도 첫 실행에는 Python 의존성 설치를 위한 인터넷 연결이 필요**합니다. 시험 파이프라인 자체는 합성 byte-token fixture를 사용하며, 유용한 언어 모델을 만들지 않습니다.

```bash
ND2_VENV=.venv031-smoke bash run_all.sh \
  --smoke --workdir "$HOME/nd2-runs/smoke031"
```

설치된 본학습 환경에서는 다음과 같이 시험할 수 있습니다.

```bash
.venv031/bin/nd2 auto --smoke --workdir "$HOME/nd2-runs/smoke031"
```

## 데이터 구성

기본 레시피는 [`nanodecision2/recipes/balanced_5m.json`](nanodecision2/recipes/balanced_5m.json)입니다.

| 분할 | 목표 판단 행 수 |
|---|---:|
| 학습 — 영어 | 2,500,000 |
| 학습 — 한국어 | 2,000,000 |
| 학습 — 기타 언어 | 500,000 |
| **학습 합계** | **5,000,000** |
| 모델 선택용 검증 | 10,000 |
| Temperature 보정 | 4,000 |
| Conformal 보정 | 4,000 |
| 최종 테스트 | 10,000 |
| **전체 합계** | **5,028,000** |

**판단 500만 행은 독립 문서 500만 개가 아닙니다.** 같은 원본의 서로 다른 질문, 서로 다른 근거 비교, 원본을 공유하는 파생 판단이 포함됩니다. 원본 기록 수·그룹 수·파생 판단 수를 별도로 보고합니다. 정제 후 수량이 부족하면 `data/shortfall.json`을 쓰고 학습을 중단하며, 복제나 임의 정답으로 채우지 않습니다.

현재 원천에는 SNLI/MultiNLI·KorNLI·XNLI·KLUE·PAWS-X·MASSIVE·NSMC·Yelp·AG News·MS MARCO·한국어 위키 근거 비교 자료가 포함됩니다. ToolACE·스팸·피싱·고객 지원 자료는 선택적 원천입니다. 자세한 식별자와 한도는 레시피, 라벨 해석과 주의점은 [원천 설명](docs/AUTO5M_SOURCES.md)을 확인하세요.

입력 초과를 조용히 잘라 정답이 보존된 것처럼 처리하지 않습니다. 기본 한도는 상태 **2,048토큰**, 질문-후보쌍 **256토큰**, 데이터 준비 단계 후보 **20개**입니다. 정확한 중복·원본 그룹 검사는 완전한 의미 중복 제거 또는 번역 품질 검수를 대신하지 않습니다.

## 학습과 모델 선택

기본 과정은 **영어 250만 행 × 1에포크**의 기준 학습 후, 해당 가중치에서 **전체 500만 행 × 2에포크**를 학습합니다. 계획상 노출은 총 **1,250만 행**이며, 서로 다른 원본 데이터가 그만큼 늘어나는 것은 아닙니다.

혼합학습은 영어 정확도/NLL과 한국어 정확도/NLL의 검증 회귀를 검사합니다. 양자화와 보정 이후에도 같은 검사로 배포 모델을 선택합니다. INT8 후보가 실패하면 FP32 후보를 확인하고, 검증된 혼합학습 개선이 없으면 기준 가중치를 보존합니다.

| `FINAL_REPORT.json`의 `quality_status` | 해석 |
|---|---|
| `mixed_candidate_passed_validation_gate` | 혼합학습 후보가 지정한 검증 조건을 통과했습니다. 최종 테스트 결과도 확인해야 합니다. |
| `reference_fallback_no_verified_mixed_improvement` | 검증된 혼합학습 개선을 확인하지 못해 영어 기준 가중치를 보존했습니다. |

이 조건은 **고정 검증셋에서의 관측 조건**입니다. 모든 영어 입력에서 성능이 절대로 떨어지지 않는다는 보장, 한·영 이해력의 동등성 또는 통계적 우월성 검정을 뜻하지 않습니다.

### 확률과 판단 보류

NLL과 Brier 손실로 확률분포를 학습하고, 점수형 문제에는 점수 순서·간격을 반영하는 누적분포 손실을 추가합니다. 최종 배포 가중치에 별도 temperature 보정을 적용하고, 분리된 conformal 데이터로 예측 집합을 구성합니다. 가능한 후보 집합이 하나로 좁혀지지 않는 경우 등에는 `abstained: true`와 `value: null`을 반환합니다.

높은 확률은 개별 정답 보증이 아닙니다. Conformal의 포함률과 보류하지 않은 응답의 정확도도 같은 개념이 아닙니다. 수식·가정·한계는 [확률 학습 설계](docs/THEORY.md)에 정리되어 있습니다.

## 모델 사용

학습이 완료되면 `WORKDIR/bundle/`을 사용합니다. 다음 예시는 실제 모델의 결과를 보장하거나 정답을 하드코딩하지 않습니다.

```python
from pathlib import Path
from nanodecision2 import DecisionEngine

engine = DecisionEngine(Path.home() / "nd2-runs/auto5m/bundle")

result = engine.decide(
    state="코드는 변경했지만 변경 이후 테스트를 실행하지 않았다.",
    questions={
        "done": {
            "type": "noul",
            "question": "이번 변경 사항의 검증이 완료되었는가?",
            "criteria": {
                "false": "변경 후 검증이 끝나지 않았다.",
                "true": "변경 후 검증을 끝냈고 통과했다.",
            },
        }
    },
    language="ko",
)

answer = result["answers"]["done"]
print(answer["probabilities"])
if answer["abstained"]:
    print("판단 보류:", answer["abstain_reasons"])
else:
    print("판단:", answer["value"])
```

실행용 설치만 필요한 경우 소스 폴더에서 `python -m pip install .`을 사용합니다. 현재 PyPI 게시를 전제로 하지 않는 **로컬 소스 설치 방식**입니다. 빌드 시 C++17 컴파일러가 필요하며, 학습용 의존성과 모델 데이터는 실행 패키지에 포함되지 않습니다.

CLI로도 사용할 수 있습니다.

```bash
.venv031/bin/nd2 predict \
  --bundle "$HOME/nd2-runs/auto5m/bundle" \
  --input examples/request.json
```

`language="ko"`는 보정 그룹을 선택하는 명시적 정보입니다. 언어를 지정하지 않으면 전역 보정을 사용합니다. 모델이 직접 외부 도구 실행이나 파일 수정 등의 작업을 수행하지는 않습니다.

## 네이티브 실행과 메모리

신경망 연산은 C++17, 원문 토큰화는 Rust 기반 `tokenizers` 바인딩을 사용합니다. Python은 텍스트 API와 패키지 인터페이스를 제공합니다. 순수 C++ CLI는 이미 토큰화한 정수 ID를 입력받습니다.

**INT8은 가중치 저장 형식이며 계산은 FP32입니다.** 가중치 양자화, 필요한 임베딩 행 복원, 메모리 매핑 및 선택적 시스템 BLAS 경로를 구현했습니다. Python을 제거하거나 INT8로 저장했다는 이유만으로 전체 메모리나 속도 개선 폭을 확정하지 않습니다. 실제 최대 RSS에는 토크나이저·활성값·캐시 등이 포함됩니다.

C++ 빌드와 별도 사용법은 [상세 실행 설명서](README_ko.md), [네이티브 소스](native)를 참고하세요.

## 중단·오류·결과 확인

중단 후에는 **같은 소스, 같은 환경, 같은 workdir**로 다시 실행합니다.

```bash
bash run_all.sh --workdir "$HOME/nd2-runs/auto5m"
```

완료 단계의 해시를 확인하고, 학습은 마지막 체크포인트에서 재개합니다. 레시피·소스·네이티브 바이너리가 바뀌면 새 workdir를 사용해야 합니다. 전원 차단 직전 저장되지 않은 일부 작업은 반복될 수 있습니다.

```text
WORKDIR/
├── recipe.lock.json          실행 설정
├── downloads/                원천 캐시와 revision 기록
├── data/                     원본 그룹·중간 풀·부족량 보고
├── cache/                    분할별 압축 토큰 캐시
├── data_report.json          실제 행·원천·그룹·용량 통계
├── english_reference/        영어 기준 학습
├── mixed_training/           혼합학습
├── reports/                  보정·검증·최종 평가
├── logs/                     단계별 로그
├── pipeline_state.json       완료 상태와 해시
├── FINAL_REPORT.json         선택 이유·실행 시간·최종 결과
└── bundle/                   사용할 실행 모델
```

| 상황 | 확인할 부분 |
|---|---|
| GPU·BF16 검사 실패 | 드라이버, CUDA용 PyTorch, WSL의 GPU 연결 상태를 확인합니다. |
| 디스크 공간 부족 | 기본 150GiB 시작 조건을 충족하는 작업 경로를 준비합니다. |
| 필수 원천 다운로드·스키마 오류 | 단계 로그와 원천 오류를 확인합니다. 접근 조건 또는 어댑터 수정이 필요할 수 있습니다. |
| 데이터 수량 부족 | `data/shortfall.json`을 확인합니다. 목표를 몰래 줄여 학습하지 않습니다. |
| 해시가 바뀌어 재개 거부 | 변경 전 환경을 복원하거나 새 workdir에서 시작합니다. |
| 기준 모델로 돌아감 | `quality_status`와 언어별 결과를 확인합니다. 학습 완료와 검증된 개선을 구분합니다. |

시간과 용량은 실행 환경·입력 길이·후보 수에 따라 달라집니다. 실행 후 보고서의 실제 단계 시간·처리량·메모리·캐시 크기를 사용하세요. 수초 단위 smoke 시간을 본학습 시간으로 해석하지 않습니다.

## 검증 범위

이번 저장소 정리 과정에서 **C++ 확장 빌드와 기존 90개 테스트를 다시 실행해 통과**했습니다. 재실행 조건과 로그는 [`docs/REPOSITORY_CHECK_2026-10-18.md`](docs/REPOSITORY_CHECK_2026-10-18.md)에 기록합니다.

이 테스트는 소형 모델·합성 자료의 코드 경로와 수치 대응을 검사합니다. 대형 가중치 다운로드, 500만 유효 행의 실제 확보, CUDA/RTX 5090 본학습, Windows 전체 경로 및 실제 언어 능력에 대한 검증을 대신하지 않습니다. 이전 배포 기록은 [`reports_v031/VERIFIED.md`](reports_v031/VERIFIED.md)에 보존되어 있습니다.

Julia 1·Laya 비교용 스크립트는 [`scripts/evaluate_raw.py`](scripts/evaluate_raw.py), [`scripts/compare_predictions.py`](scripts/compare_predictions.py)입니다. 같은 문항·질문·후보 조건과 실패율을 함께 비교하는 방식이며, 자세한 조건은 [벤치마크 프로토콜](docs/BENCHMARK_PROTOCOL.md)을 참고하세요. 실제 세 모델 가중치를 사용한 비교와 별도 MLM 추가학습은 기본 자동 파이프라인에 포함되지 않습니다.

테스트를 직접 실행하려면 설치된 환경에서 다음 명령을 사용합니다.

```bash
.venv031/bin/python -m pytest -q
```

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `run_all.sh`, `run_all.ps1` | Linux/WSL 자동 실행기와 PowerShell 보조 실행기 |
| `nanodecision2/` | 데이터 준비·학습·평가·확률 보정·Python API |
| `nanodecision2/recipes/` | 기본 Auto5M 레시피 |
| `native/` | C++17 추론 엔진·Python 확장·CLI |
| `scripts/` | 선택적 데이터 처리·인코더 추가학습·외부 모델 비교 |
| `tests/` | 단위·통합시험 |
| `docs/` | 확률 이론·데이터 원천·벤치마크·이전 버전 설명 |
| `examples/` | 요청 형식과 개발용 점검 예제 |

## 라이선스와 출처

루트의 [Apache-2.0 `LICENSE`](LICENSE)는 그대로 유지합니다. 별도 `LICENSES/` 폴더 대신 가져온 NanoDecision2 소스의 **MIT 저작권·허락 고지 전문은 [`NOTICE`](NOTICE)에 보존**했습니다. Python 패키지 메타데이터의 MIT 표기도 유지합니다.

외부 인코더·토크나이저·데이터셋은 각 원천의 이용 조건을 따릅니다. 자동 다운로드가 외부 데이터의 재배포 권한을 일괄 부여하지는 않습니다. 학습 데이터·모델 가중치·로컬 인증 정보는 이 저장소에 포함하지 않습니다.
