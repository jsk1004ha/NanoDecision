# NanoDecision2 0.3.1 — Auto5M

공개 데이터 다운로드 → 정제·토큰화 → **학습용 5,000,000개 판단 문제** 구성 → 영어 기준 모델 학습 → 한·영·다국어 혼합학습 → 내보내기 → 확률 보정 → 언어별 채택 검사 → 최종 평가를 연결합니다. Jev API나 유료 교사 API를 호출하지 않습니다. 실행 모델은 기존 C++17 엔진과 Rust 토크나이저/Python 인터페이스를 유지합니다.

**이 배포물은 코드이며, 실제 500만 행으로 학습을 끝낸 가중치가 아닙니다.** 소형 오프라인 통합시험과 단위시험은 실행했습니다. 이 제작 환경에서는 외부 대형 다운로드와 CUDA 학습을 실행하지 못했습니다. 따라서 모든 원천의 현재 다운로드·스키마, 실제 500만 유효 행 확보, RTX 5090 메모리/속도, Julia/Laya 초과 성능은 아직 실측 검증되지 않았습니다. 오류를 숨기거나 수량을 줄여 성공으로 표시하지 않습니다.

## 1. 실행 — Linux/WSL Ubuntu

ZIP을 풀고 `run_all.sh`가 있는 `NanoDecision2` 폴더에서 실행합니다.

```bash
bash run_all.sh --workdir "$HOME/nd2-runs/auto5m"
```

이 명령이 가상환경 생성, 고정 버전 Python 의존성 설치, CUDA PyTorch 설치, 네이티브 확장 빌드, 전체 파이프라인을 연결합니다. 이미 설치가 끝난 동일 소스에서는 설치를 반복하지 않습니다. 기본 가상환경은 `.venv031`입니다. 기존 환경을 덮어쓰지 않습니다.

필요한 OS 조건은 Python **3.11~3.13**, C++17 컴파일러, Python 개발 헤더, CUDA를 사용할 수 있는 NVIDIA 드라이버, 인터넷 연결입니다. Ubuntu에 기본 개발 도구가 없다면 **처음 한 번** 아래 명령이 필요합니다. 실행기가 운영체제나 GPU 드라이버를 몰래 변경하지는 않습니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libopenblas-dev
```

자동 본학습은 CUDA/BF16을 요구하며, GPU가 안 보일 때 CPU로 조용히 전환하지 않습니다. 실제 GPU 행렬 계산까지 검사합니다. 설치 프로필은 `torch==2.10.0`의 공식 CUDA 12.8 휠입니다. CUDA toolkit을 별도로 설치하는 작업은 자동 실행에 포함하지 않습니다. 휠과 호환되는 드라이버가 필요합니다. `torchvision`/`torchaudio`는 설치하지 않습니다.

기본 실행 전 **작업 디스크 150GiB 이상 여유 공간**을 검사합니다. 실제 사용량의 실측 예측이 아니라 다운로드·SQLite 중간 풀·압축 캐시·여러 체크포인트를 위한 보수적인 예산입니다. 데이터 길이/원천 배포 형식에 따라 초과할 수 있습니다. SSD, 시스템 RAM 32GB 이상을 운영상의 출발 조건으로 권장하지만 대형 학습의 메모리 사용을 실측한 값은 아닙니다. WSL에서는 작업 디렉터리를 Linux 파일시스템 안에 두는 구성을 권합니다.

중단/재부팅 후에는 **같은 소스, 같은 환경, 같은 명령**을 다시 실행하세요. 완료 단계는 파일 해시를 확인하고 건너뜁니다. 학습은 마지막 체크포인트부터 계속합니다. 정기 저장 이후 전원 차단까지의 작업은 다시 수행될 수 있습니다. 정상 Ctrl+C는 현재 옵티마이저 단계가 끝날 때 저장하도록 요청합니다.

Windows PowerShell에서 기본 WSL 배포판을 호출하는 보조 실행기도 있습니다.

```powershell
powershell -File .\run_all.ps1 --workdir /home/YOUR_USER/nd2-runs/auto5m
```

WSL 설치/드라이버 설치는 별도입니다. PowerShell 래퍼와 실제 Windows/CUDA 조합은 제작 환경에서 실행 검증하지 못했습니다.

## 2. 무엇이 자동으로 실행되는가

| 순서 | 처리 |
|---|---|
| 1 | 환경·GPU·디스크·네이티브 확장 확인 |
| 2 | mmBERT-base revision 고정, 가중치·토크나이저 다운로드, 원본 인코더와 출력 일치 검사 |
| 3 | 공개 원천을 shard 단위로 내려받고 라벨 의미를 보존해 판단 문제로 변환 |
| 4 | 길이/후보/라벨 검사, 중복·원본 그룹 정리, 분할 오염 검사, 정확한 언어별 수량 선택 |
| 5 | 디스크 압축 토큰 캐시 생성 |
| 6 | 같은 학습 풀의 영어 250만 행으로 영어 기준 모델 1에포크 학습 |
| 7 | 그 모델에서 시작해 전체 500만 행 2에포크 혼합학습 |
| 8 | 영어 정확도/NLL 및 한국어 정확도/NLL 조건을 만족하는 체크포인트만 후보로 채택 |
| 9 | 기준 FP32·후보 INT8 내보내기, 실제 배포 가중치의 확률 보정 |
| 10 | 배포 형태에서도 언어별 조건 확인; INT8 실패 시 후보 FP32, 다시 실패하면 기준 모델 보존 |
| 11 | 선택이 끝난 모델을 최종 테스트에서 평가하고 `bundle/`, `FINAL_REPORT.json` 저장 |

중간 모델 검증은 고정된 언어·과제별 부분집합 최대 6,000행을 사용합니다. 매 10,000 옵티마이저 단계와 에포크 끝에 검증합니다. 모든 단계마다 대규모 검증 전체를 돌리는 구조가 아닙니다. 기준/후보/배포 비교에 같은 부분집합을 사용합니다. 최종 테스트는 모델 선택에 사용하지 않습니다.

기본 batch 2 × gradient accumulation 32 = 유효 batch 64입니다. 영어 기준 학습까지 포함하면 계획상 학습 노출은 **250만 + 500만×2 = 1,250만 행**입니다. 1,250만 개의 새 독립 데이터가 있다는 뜻이 아닙니다. BF16과 gradient checkpointing은 학습에 사용하며 배포는 CPU C++입니다.

## 3. 실제 수량 정의

| 분할 | 행 수 |
|---|---:|
| **train** | **5,000,000** |
| validation | 10,000 |
| temperature | 4,000 |
| conformal | 4,000 |
| test | 10,000 |
| **전체** | **5,028,000** |

학습용은 영어 **2,500,000**, 한국어 **2,000,000**, 기타 언어 **500,000**입니다. 이 비율은 최적 성능이 입증된 배합이 아니라 명시적인 초기 레시피입니다. 예제별 신뢰도 가중치가 달라 실제 gradient의 언어 비율과 행 비율이 완전히 같지는 않습니다. 노출량과 유효 가중치를 기록합니다.

이전 대화의 550만 행은 설계 예산이었습니다. **이번 기본 자동화의 실제 예약량은 502만8천 행**으로, 학습 500만 행을 유지하면서 검증/보정 운영 비용을 줄였습니다. JSON 레시피에 명시되어 있으며 몰래 수량을 변경하지 않습니다.

**5,000,000행은 서로 다른 입력·질문·후보로 정의된 판단 문제 수이지, 독립 문서 500만 개가 아닙니다.** 원본 대화의 다른 시점, 동일 질의의 다른 근거 비교, NLI의 별도 함의 여부 질문 등이 포함됩니다. `train_original_records`, `train_source_groups`, `train_derived_decisions`를 따로 기록합니다. 원본 기록 수 역시 출처 ID 기준이며, 언어를 넘어 완전한 의미 중복 제거를 마친 독립 표본 수는 아닙니다.

정제 후 언어별 수량이 부족하면 `data/shortfall.json`을 쓰고 학습을 중단합니다. 같은 행 복제, 무의미한 접미사, 임의 정답, 허위 레코드로 채우지 않습니다. `--smoke`의 합성 fixture는 명시적인 개발시험용 예외이며 본학습 경로에서는 절대 사용하지 않습니다.

## 4. 데이터 원천과 변환

기본 레시피는 `nanodecision2/recipes/balanced_5m.json`입니다. 현재 선택한 원천은 다음과 같습니다.

| 원천 | 의미와 용도 |
|---|---|
| SNLI/MultiNLI, KorNLI, XNLI | 함의·중립·모순. 번역 원천과 원본 관계를 구분 |
| KLUE NLI/YNAT | 한국어 문장 관계·제목 주제 |
| PAWS-X 7개 언어 | 의미가 같은지, 표면 표현과 의미의 차이 |
| MASSIVE 7개 locale | 18개 상황 전체 후보 분류 |
| NSMC / Yelp full review | 한국어 긍정 여부 / 영어 리뷰 별점 Score |
| AG News | 영어 뉴스 주제 |
| MS MARCO v2.1 | 질문-문서의 선택된 근거 여부. 약한 selection label |
| ko-wiki-reranking | 공개 positive와 서로 다른 mined negative의 근거 비교 |
| ToolACE, Enron spam, phishing, support tickets | 선택적 도구 호출·스팸·피싱·담당 부서 데이터 |

KorNLI는 공식 제작자의 TSV를 직접 받도록 연결했습니다. 그 외 Hub 원천은 Python dataset script 실행 대신 실제 Parquet/JSON 파일을 찾습니다. 필요하면 Hub의 converted-Parquet revision을 별도로 고정합니다. 모델 API 추론 비용은 없습니다. 원천 이용 조건/접근 정책은 각 제공자의 조건을 따릅니다. gated xLAM이나 미연결 Nemotron 전체 다운로드는 기본 레시피에 넣지 않았습니다.

한국어 수량을 구성하기 위해 ko-wiki-reranking의 **서로 다른 negative**를 최대 12개까지 활용하고, 낮은 우선순위로 KorNLI의 별도 binary 함의 판단을 추가합니다. 예컨대 NLI가 neutral이면 `가설이 거짓이다`가 아니라 `전제로부터 가설이 반드시 참이라고 결론내릴 수 없다`를 정답으로 사용합니다. 같은 정답 문제의 문구만 바꿔 복제하는 방식이 아닙니다. 그래도 원본을 공유하므로 독립 표본으로 과장하지 않습니다.

MS MARCO의 미선택 passage를 “실제로 반드시 무관련”이라고 라벨링하지 않습니다. ToolACE는 공개된 합성 호출의 보수적인 단일 도구 예제만 변환하고, 정답인 현재 assistant 호출을 state에 넣지 않습니다. ko-wiki의 mined negative와 공개 합성/번역 자료도 인간이 검증한 진실과 동일하게 취급하지 않고 출처·가중치를 남깁니다.

필수 원천에 접근하지 못하거나 스키마가 맞지 않으면 중단합니다. 선택적 원천 실패는 `source_errors.json`에 남기며, 이 경우에도 최종 정확한 수량 조건은 완화하지 않습니다. 원천마다 최대 수량/우선순위가 있어 모든 원문을 무한히 내려받지는 않습니다. 최종 과제별 비율은 `by_task`, 원천별 비율은 `by_source`에 남깁니다. 종전 제안의 과제별 목표 배분이 모두 충족되었다고 가정하지 않습니다. 특히 완료/실패 로그의 풍부한 실제 사례를 이 원천들만으로 다 확보한 것은 아닙니다.

## 5. 전처리·분할·재개

- 원문에서 조사·부정·조건을 삭제하지 않습니다. 입력 초과는 원문/후보를 조용히 자르지 않고 사유를 기록해 제외합니다. 기본 state 2,048토큰, 질문-후보쌍 256토큰, 후보 최대 20개입니다.
- 원본 그룹과 Unicode 정규화한 동일 상태의 연결 성분을 합칩니다. 서로 충돌하는 라벨은 제외합니다. 완전한 의미/근접 중복 제거 알고리즘은 아닙니다.
- 같은 PAWS/MASSIVE 원본의 언어별 번역은 같은 그룹입니다. 검증되지 않은 NLI 번역 정렬 때문에 영어/한국어/다국어 NLI 원본 train은 모두 train-only로 유지합니다.
- 공식 test/development와 겹치는 그룹은 학습에서 제외합니다. 기타 원본 train 그룹은 고정 hash로 94/2/2/2 분할한 후 목표 수량을 선별합니다. 공식 test는 model selection에 사용하지 않습니다.
- SQLite에 shard hash, 다음 원본 행 위치, 수집/제외 사유, 그룹 정보를 저장합니다. 준비된 토큰 캐시는 한 레코드씩 zlib 압축합니다. 모든 원문/토큰을 Python 리스트로 한꺼번에 올리지 않습니다.
- HTTP 직접 다운로드는 ETag/Range 재개, Hub는 revision 고정과 캐시를 사용합니다. 완료 모델·캐시·평가 결과의 해시가 바뀌면 재개를 거부합니다. 장애 시 최근 저장 이후 일부 읽기/연산은 반복될 수 있습니다.
- 레시피/소스 코드/네이티브 바이너리가 달라지면 새 workdir가 필요합니다. 실험 설정이 달라진 것을 정확한 재개로 가장하지 않습니다.

## 6. 영어 보존 조건의 의미

혼합학습 전 같은 구조로 영어 기준 모델을 실제 학습합니다. 후보가 그 기준보다 영어 정확도가 낮거나 NLL이 높으면 best 후보로 채택하지 않습니다. 한국어 정확도/NLL도 악화되어서는 안 됩니다. 이 조건을 통과한 후보 중 macro-language NLL이 개선되는 모델을 남깁니다. 양자화/보정 이후에도 같은 검사를 다시 합니다.

이는 **해당 고정 검증셋에서의 관측 조건**입니다. 모든 영어 입력에서 성능이 절대 떨어지지 않는다는 보장이나 통계적 우월성 검정이 아닙니다. 한국어가 영어와 동등해졌다거나 Julia/Laya를 넘었다는 뜻도 아닙니다.

`FINAL_REPORT.json`의 `quality_status`를 확인하세요.

- `mixed_candidate_passed_validation_gate`: 혼합학습 후보가 지정된 검증 조건을 통과했습니다. 최종 테스트 결과를 함께 확인해야 합니다.
- `reference_fallback_no_verified_mixed_improvement`: 후보 개선을 확인하지 못해 영어 기준 가중치를 보존했습니다. 학습 완료를 한국어 성능 향상으로 표시하지 않습니다. 기준 가중치의 INT8 표현이 조건을 통과하면 `selected_format: int8`일 수도 있습니다.

## 7. 출력과 모델 사용

```text
WORKDIR/
  recipe.lock.json                  사용한 설정
  model_revision.lock.json         모델 revision
  downloads/sources.lock.json      데이터 revision/파일 목록
  data/pool.sqlite                 원본·그룹·토큰화 중간 풀
  cache/                          분할별 압축 token cache와 JSONL.gz
  data_report.json                정확한 행·언어·원천·파생 수량/캐시 바이트
  english_reference/              영어 기준 모델/checkpoint
  mixed_training/                 혼합학습 모델/checkpoint
  exports/                        기준 FP32, 후보 INT8, 필요하면 FP32
  reports/                        보정·검증·최종 평가
  bundle/                         최종 선택된 실행 모델
  logs/                           단계별 로그
  pipeline_state.json             완료/실패 단계와 해시
  FINAL_REPORT.json               모델 선택 이유·언어별 결과·단계 시간·용량
```

실행 완료 후 같은 가상환경에서:

```bash
.venv031/bin/python - <<'PY'
from pathlib import Path
from nanodecision2 import DecisionEngine
engine = DecisionEngine(Path.home() / 'nd2-runs/auto5m/bundle')
result = engine.decide(
    state='코드는 변경했지만 변경 이후 테스트를 실행하지 않았다.',
    questions={'done': {
        'type':'noul',
        'question':'이번 변경 사항의 검증이 완료되었는가?',
        'criteria':{'false':'변경 후 검증이 끝나지 않았다.',
                    'true':'변경 후 검증을 끝냈고 통과했다.'}
    }},
    language='ko',
)
print(result)
PY
```

이는 인터페이스 예시이지 이 문항에 대한 정확도 보증이 아닙니다. 최종 경로를 바꿨으면 Python의 경로도 맞추세요. 배포만 할 때는 런타임 패키지와 `bundle`만 필요합니다. 신경망 추론은 C++, 토크나이저는 Rust 바인딩이며 PyTorch는 추론 경로에서 필요하지 않습니다.

INT8은 가중치 저장 형식이며 연산은 FP32입니다. 자동 확률 보정/평가는 배포 파일의 양자화 가중치를 정확히 복원한 PyTorch FP32 모델을 사용해 GPU에서 배치 처리하고, 실제 C++ 출력과 표본 일치 검사를 수행합니다. 이를 전체 C++ 처리량 실측으로 표시하지 않습니다. 샘플 일치 검사가 실패하면 배포 과정을 중단합니다.

## 8. 다른 실행 방법

실제 다운로드·학습 없이 설치 이후 통합시험:

```bash
.venv031/bin/nd2 auto --smoke --workdir "$HOME/nd2-runs/smoke031"
```

완전히 별도의 CPU 개발 환경을 만들 때는 `ND2_VENV=.venv031-smoke bash run_all.sh --smoke`를 사용할 수 있습니다. 본학습용 CUDA venv를 smoke CPU용으로 바꾸지 않도록 환경을 분리하세요. smoke의 모든 가중치는 `smoke_only: true`이고 실제 판단 모델이 아닙니다.

다운로드·500만 행 구성까지만 먼저 완료:

```bash
bash run_all.sh --workdir "$HOME/nd2-runs/auto5m" --stop-after-stage data
# 이후 동일 workdir에서 --stop-after-stage를 빼면 학습부터 진행
```

OS 환경을 직접 관리하는 경우:

```bash
# 공식 CUDA 휠은 별도 설치. PyPI에 게시된 패키지가 아니라 로컬 소스입니다.
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[auto]'
nd2 auto --workdir runs/auto5m
```

## 9. 범위와 검증

본학습의 실제 시간은 해당 컴퓨터의 데이터 길이·GPU 처리량을 확인해야 합니다. 0.3.1은 실제 단계 경과 시간, 학습 처리량, CUDA 최대 할당/예약 메모리, 선택한 캐시의 바이트 수, 작업 디렉터리 논리 바이트를 보고합니다. 작업 디렉터리 논리 바이트는 압축/파일시스템 할당 블록 수와 다릅니다. 실행 전에 며칠이라고 확정하지 않습니다.

이 명령에 **별도 1억 토큰 MLM 추가학습**은 넣지 않았습니다. 기존 `scripts/adapt_encoder.py`는 별도 선택 기능으로 유지합니다. Julia/Laya 실가중치 다운로드·실제 3자 벤치마크도 자동 본학습의 필수 단계가 아니며 기존 비교 스크립트를 유지합니다. 이전의 모든 설계 제안이 자동으로 실행되는 것처럼 설명하지 않습니다.

`reports_v031/VERIFIED.md`에 실제 0.3.1 시험 범위와 실행하지 못한 부분을 적었습니다. `reports/`, `reports_v030/`, `docs/LEGACY_0.3.0_README.md`는 역사적 기록이며 새 성능 수치로 사용하면 안 됩니다.

코드는 MIT, 외부 모델/데이터는 각 원천 조건을 따릅니다. 자동 수집이 모든 원천에 일괄적인 재배포 권한을 부여하는 것은 아닙니다.

## 참고 원천

- 공식 PyTorch 버전별 휠: https://pytorch.org/get-started/previous-versions/
- mmBERT: https://huggingface.co/jhu-clsp/mmBERT-base
- KorNLI 원저자: https://github.com/kakaobrain/kor-nlu-datasets
- PAWS-X: https://huggingface.co/datasets/google-research-datasets/paws-x
- MASSIVE: https://github.com/alexa/massive
- MS MARCO: https://huggingface.co/datasets/microsoft/ms_marco
- 한국어 mined-negative 자료: https://huggingface.co/datasets/upskyy/ko-wiki-reranking
- ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE
- Yelp: https://huggingface.co/datasets/Yelp/yelp_review_full
- Hugging Face 파일 다운로드/캐시: https://huggingface.co/docs/huggingface_hub/guides/download
