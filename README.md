# NanoDecision

**질문·판정 기준·후보를 읽고 최선의 판단과 확률을 반환하는 로컬 결정 모델.**

Python 학습 코드와 C++17 추론 엔진, Rust 기반 토크나이저 바인딩, `nd2` CLI를 제공합니다. Jev API 또는 유료 교사 API는 필수 의존성이 아닙니다.

> **이 브랜치는 학습 수정 계획 v1.2의 1차 구현입니다.** 출력 계약 v2, 데이터 수집 오류 복구, 병렬 추론/캐시 로더, 검토된 지시문 쌍 학습을 포함합니다. 새로운 학습 가중치나 0.4.0 전체 구현을 제공하는 것은 아닙니다. 센서 인코더·장문 재학습·JevBench 연동 등 남은 항목은 [구현 범위와 이행 안내](docs/changes/PLAN_V1_2_IMPLEMENTATION.md)에 명시했습니다.

[변경 및 사용법](docs/changes/PLAN_V1_2_IMPLEMENTATION.md) · [이번 검증](docs/changes/VALIDATION_V1_2_FOUNDATION.md) · [기존 한국어 벤치마크](benchmark.ko.md) · [English benchmark](benchmark.md)

## 기본 판단 계약

- **Choice:** 주어진 후보 전체의 확률과 최고 확률 후보 ID를 반환합니다.
- **Noul:** 참/거짓 분포, 참일 확률, 최선의 Boolean 판단을 반환합니다.
- **Score:** 레벨별 확률, 최빈 레벨, 기대 점수를 구분합니다. `value`는 기존과 같은 기댓값입니다.

정상 입력은 확률이 낮거나 기존 calibration에 `qhat`가 있다는 이유로 `null`이 되지 않습니다. 모델이 요청하지 않은 ‘모름’ 후보를 추가하지 않으며, NLI 중립 같은 원래 라벨은 보존합니다. 확률을 인위적으로 높이지 않습니다. 파일 손상·잘못된 후보·지원 한도 초과는 오류입니다.

```python
from nanodecision2 import DecisionEngine
engine = DecisionEngine("bundles/trained")
result = engine.decide(
    state="코드를 수정했지만 변경 후 tests는 실행하지 않았다.",
    questions={"done": {
        "type": "noul", "question": "변경 후 테스트를 실행했는가?",
        "criteria": {"false": "아직 실행하지 않음", "true": "실행함"},
        "code_switch": True,
    }},
)
print(result["answers"]["done"]["value"])
print(result["answers"]["done"]["probabilities"])
```

예제는 인터페이스 설명이며 정확도를 보장하지 않습니다. `bundles/trained`는 사용자가 학습하거나 확보한 번들입니다. 낮은 확률의 결과를 실제 외부 행동에 사용할지는 별도 애플리케이션 정책입니다. 구형 자동 보류가 필요한 앱만 `legacy_policy=True`를 명시합니다. 시험 모델은 `allow_smoke=True`가 필요하며 `status=test_only`로 표시됩니다.

## 시작하기

Linux/WSL Ubuntu의 Python 3.11–3.13과 C++17 개발 도구, 학습용 NVIDIA 드라이버가 필요합니다. 최초 원천·모델 다운로드에는 인터넷이 필요합니다. GPU가 없으면 대규모 학습이 CPU로 조용히 전환되지는 않습니다.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential libopenblas-dev
git clone https://github.com/jsk1004ha/NanoDecision.git
cd NanoDecision
# PR 검토 시 해당 PR 브랜치를 checkout한 다음 새 작업 폴더 사용
bash run_all.sh --workdir "$HOME/nd2-runs/v12-foundation"
```

설치기는 로컬 가상환경과 네이티브 확장을 구성하며 운영체제·NVIDIA 드라이버를 변경하지 않습니다. 기본 학습 레시피는 디스크 여유 150GiB를 검사하지만, 이는 실측 사용량이 아닌 시작 예산입니다.

```text
환경 검사 → 원천 경로·파서 샘플 검사 → 백본 초기화
→ 다운로드·정제·토큰 캐시 → 영어 기준 학습 → 다국어 혼합학습
→ 배포 가중치 내보내기 → 온도 보정 → 검증·최종 평가 → bundle/
```

기본 목표는 영어 250만·한국어 200만·기타 50만의 **학습 판단 500만 행**입니다. 검증 1만·온도 보정 4천·새 최종 테스트 1만4천을 분리합니다. 독립 원본 500만 개라는 뜻이 아니며 부족하면 학습을 차단합니다. 전체 실제 다운로드·GPU 학습의 성공과 Julia/Laya 초과 성능을 이번 PR에서 검증하지 않았습니다.

**기존 실행에서 새 코드로 바로 exact resume하지 마세요.** 소스·파서 계약 변경에 따라 변환 캐시와 실험 지문을 다시 만들어야 합니다. 기존 체크포인트는 데이터 감사를 거쳐 warm start로 활용할 수 있습니다. 같은 버전·환경·workdir로 중단한 실행만 같은 명령으로 재개합니다.

## 원천 검사 및 지시문 학습

TSV 따옴표를 삭제하지 않고 원천별 인용 규칙을 명시합니다. SNLI `plain_text` 경로를 처리하며, 한국어 위키 기본 소스는 공개 `data` 구성으로 명시적으로 변경합니다. 이를 과거 mined-negative JSONL과 동일하다고 설명하지 않습니다. 명시한 파일 variant가 없으면 다른 자료로 몰래 대체하지 않습니다.

```bash
python -m nanodecision2.source_preflight \
  --recipe nanodecision2/recipes/balanced_5m.json \
  --workdir "$HOME/nd2-runs/source-probe" --samples 3
```

검토한 `instruction_views`가 있는 JSONL은 아래 도구로 기존 JS 일관성 학습에 연결합니다. 학습 입력이 별도로 필요하며 임의 변형을 자동 승인하지 않습니다.

```bash
python -m nanodecision2.instruction_views --input data/reviewed.jsonl \
  --out data/instruction-pairs.jsonl --fraction 0.25
# 준비된 캐시로 학습할 때 사용 가능한 옵션
# --pair-consistency 0.05 --selection-metric macro_task_error --workers 2 --prefetch 2
```

## 작은 통합시험과 패키지

```bash
# 실행 전용 로컬 소스 설치
python -m pip install .
# 학습과 CPU 테스트 의존성이 필요할 때
python -m pip install ".[train,verify,data,test]"
python -m pytest -q
nd2 auto --smoke --workdir runs/smoke-v12
```

smoke는 byte-token 소형 무작위 모델입니다. 실제 언어 성능을 뜻하지 않습니다. PyPI 등록이나 대형 학습 가중치 배포를 전제로 하지 않습니다. 실배포 시 모델 bundle과 실행 의존성만 필요하며 추론은 torch/NumPy/Transformers를 import하지 않습니다.

C++ 가중치는 INT8로 저장할 수 있지만 계산은 FP32입니다. 새 병렬 경로의 처리량·RSS·CUDA 속도 향상 폭은 실제 장치에서 측정해야 합니다. 모델 포맷은 유지하며 새 응답의 `api_contract_version`은 2입니다. 패키지 릴리스 번호와 API 계약 버전은 별개입니다.

## 벤치마크와 문서

[benchmark.ko.md](benchmark.ko.md)와 [benchmark.md](benchmark.md)는 기존 모델의 결과입니다. 이번 코드 변경의 새 정확도·속도 측정값으로 해석하지 마세요. 공개 JevBench의 실제 어댑터·원 채점기 연동은 후속 작업이며 공식 전체 순위를 주장하지 않습니다.

[README_ko.md](README_ko.md)는 **0.3.1의 역사적 상세 안내**입니다. 기본 보류·conformal·데이터 원천 선택에 관한 동작은 이 README와 [v1.2 이행 문서](docs/changes/PLAN_V1_2_IMPLEMENTATION.md)가 우선합니다. 기존 테스트·검증 기록은 소형/대형과 실측/미검증을 구분해 보존합니다.

## 라이선스

루트 [LICENSE](LICENSE)와 [NOTICE](NOTICE)는 변경하지 않았습니다. 외부 백본·토크나이저·데이터의 조건은 각 원천을 따릅니다. 원문 데이터와 모델 가중치, 인증정보는 이 PR에 포함하지 않습니다.
