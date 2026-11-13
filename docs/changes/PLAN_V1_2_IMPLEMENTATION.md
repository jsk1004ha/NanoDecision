# v1.2 학습 계획 — 1차 구현 및 이행 안내

2026-11-13 · 기반 커밋 `7c6c03d55d9522129efb89e7c278c13f89429011`

이 PR은 사용자가 확정한 **「NanoDecision 학습 수정 계획서 v1.2」의 P0 및 일부 P1/P2 기반 작업**입니다. 계획 전체 구현이나 새로운 학습 가중치의 출시가 아닙니다. 원본의 사적 Drive 링크·원문 CSV·모델 가중치는 업로드하지 않습니다. 기존 LICENSE/NOTICE와 별도 대시보드 PR은 변경하지 않습니다.

원본 계획서 Markdown SHA-256: `2d6bd9d3604e18f6f797531707b43bc7cd2b3b0b693cd0e5b935afa10a545515`. 이 파일은 원본 계획의 공개 구현 추적 문서이며 원본 34쪽 전체를 재작성한 파일은 아닙니다.

## 교정된 목표

주어진 입력·질문·판정 기준·후보를 잘 구분해 **최선의 판단과 확률을 반환**합니다. 답의 존재 여부를 먼저 판정하는 공통 게이트, 자동 `unknown` 후보, 낮은 확률에 따른 기본 `null`을 사용하지 않습니다. NLI 중립처럼 과제에서 정의한 라벨은 그대로 보존합니다. 학습 정답을 검증하는 오프라인 검사는 계속 필요합니다.

## 이 PR에서 구현한 항목

| 항목 | 구현 / 범위 |
| --- | --- |
| 출력 계약 v2 | 정상 입력의 Choice/Noul/Score 결과는 non-null. `qhat`가 있는 이전 보정 파일도 기본 결정을 지우지 않음 |
| 외부 정책 분리 | `min_confidence`는 advisory `policy_result`만 생성. 이전 null 동작은 명시적인 `legacy_policy=True` |
| 결과 의미 | `selected_option_id`, `top1_probability`, `tied_candidate_ids`, Score `mode_value`와 `expected_value` 분리. 동점은 ID 정렬 |
| 파일 무결성 | 보정 유무와 관계없이 모델 파일 SHA 확인. 토크나이저 provenance가 있으면 SHA 확인 |
| 시험 모델 구분 | fixture 번들은 `allow_smoke=True`를 명시해야 실행. 결과 `status=test_only` |
| 언어 전달 | Python/predict/stdin의 요청·질문별 언어 전달. 선언된 본문·질문·후보 언어가 다르면 `mixed` 온도 버킷, 없으면 전역값 |
| TSV 복구 | 원천별 dialect·열 수·헤더·인코딩 검사. KorNLI만 literal-quote TSV로 명시. quoted 탭/개행을 사용하는 다른 원천은 기존 문법 유지 |
| 원천 경로 복구 | SNLI `plain_text`, 구성별 Parquet/JSONL 탐색과 선언된 data_files 메타데이터. 명시적 파일 조건을 fallback에서 버리지 않음 |
| 한국어 위키 | 기본 레시피는 현재 공개 `data` 구성을 **명시적으로 선택**. 과거 `train_minedHN.jsonl`과 동일 자료라고 간주하지 않음 |
| 원천 사전검사 | 모델 초기화 전에 첫 shard의 소수 원본·변환 라벨 검사. 전체 코퍼스 감사나 다운로드 완료를 뜻하지 않음 |
| 병렬 추론 | 전체 모델 잠금 대신 캐시 조회/게시/삭제만 잠금. 읽기 전용 가중치 공유, 계산 버퍼는 요청별 독립 |
| 병렬 로더 | `--workers`, `--prefetch`; spawn worker가 캐시 해제/읽기를 담당. collate·후보 셔플은 학습 프로세스에서 수행 |
| RoPE | 같은 forward에서 레이어별 동일 삼각함수 계산 재사용. 가중치/네이티브 파일 형식 유지 |
| 지시문 학습 입력 | 검토한 동일 의미의 질문·후보 변형으로 정확히 2개 view를 구성. 같은 언어의 패러프레이즈도 기존 JS 경로에서 지원 |
| 모델 선택 | `macro_task_error`(1 − 과제별 평균 정확도) 우선, 동률은 NLL. 기존 영어·한국어 보호 조건 유지 |
| 기본 보정 | temperature-only. 기본 자동 파이프라인에서 conformal 단계를 제거; 명시적 레거시 보정은 별도 지원 |

**동등 의미는 코드가 자동으로 증명하지 않습니다.** `reviewed=true`는 데이터 검토자의 선언입니다. 후보 의미·부정·주체·기준이 달라진 문제는 같은 일관성 쌍으로 넣지 않습니다.

## 출력 API 이행

```python
from nanodecision2 import DecisionEngine
engine = DecisionEngine("bundles/trained")
result = engine.decide(
    state="코드 수정 후 tests는 아직 실행하지 않았다.",
    questions={"done": {
        "type": "noul", "question": "Have the post-change tests run?",
        "criteria": {"false": "아직 실행하지 않음", "true": "실행함"},
        "state_language": "mixed", "question_language": "en",
        "candidate_languages": {"false": "ko", "true": "ko"},
    }},
)
answer = result["answers"]["done"]
print(answer["value"], answer["probabilities"], answer["api_contract_version"])
```

`require_calibration=True`나 `min_confidence=0.9`를 명시해도 새 기본 모드에서는 모델 결정을 보존하고 정책 경고만 제공합니다. 이전처럼 `value=None`이 필요한 사용자는 `DecisionEngine(..., legacy_policy=True)`를 명시해야 합니다. 파일 손상·지원하지 않는 길이·잘못된 후보는 여전히 오류입니다. 시험 모델은 별도로 `allow_smoke=True`를 요구합니다. 새로운 입력 언어 전달은 확률 보정 경로 수정이지, 재학습 없이 혼합언어 정확도가 올라갔다는 뜻은 아닙니다.

```bash
nd2 predict --bundle bundles/trained --input examples/request.json
# 구형 동작이 필요한 별도 앱만:
nd2 predict --bundle bundles/trained --input examples/request.json --legacy-policy
# 명시적 시험 모델만:
nd2 predict --bundle runs/fixture/bundle --input examples/request.json --allow-smoke
```

Score의 `value`는 기존과 같은 기댓값이며 후보 레벨 중 하나일 필요가 없습니다. 모달 레벨은 `mode_value`, 최빈 후보 확률은 `top1_probability`입니다. 그 확률을 기댓값의 신뢰구간으로 해석하지 않습니다.

## 데이터 복구와 실행

**새 작업 디렉터리를 사용하세요.** 파서·resolver·코드가 바뀐 실행을 기존 캐시의 exact resume으로 표시하지 않습니다. 원본 다운로드는 파일 해시를 검증한 캐시만 재사용할 수 있으며, decoded/token 캐시와 데이터 품질은 새로 검사해야 합니다. 잘못 읽힌 데이터로 학습한 모델은 코드 수정만으로 복구되지 않습니다.

```bash
# GPU/모델 다운로드 없이 원천의 첫 shard와 샘플 검사
python -m nanodecision2.source_preflight \
  --recipe nanodecision2/recipes/balanced_5m.json \
  --workdir "$HOME/nd2-runs/source-probe-v12" --samples 3

# 설치·다운로드·학습은 새 workdir에서. 데이터/학습은 실제로 비용이 듭니다.
bash run_all.sh --workdir "$HOME/nd2-runs/v12-foundation"
```

사전검사는 실제 첫 shard 파일을 다운로드하므로 원천에 따라 다운로드가 작지 않을 수 있습니다. 검사는 첫 샘플에 한정되며, 나머지 행의 오류는 전체 전처리에서 계속 검사합니다. 대형 원천·실제 GPU 경로를 이번 제작 환경에서 검증하지 않았습니다.

TSV의 `parser`에는 quoting(`none`/`minimal`), delimiter, expected_columns, encoding, quotechar, escapechar, doublequote, field_size_limit를 선언할 수 있습니다. `none`은 원문 따옴표를 보존하는 원천에만 사용합니다. 열 오류를 조용히 skip하거나 다른 dialect를 시도해 우연히 읽힌 결과를 채택하지 않습니다.

한국어 위키의 기본 소스는 `config=data`, `variant=published_data_config`입니다. 원천의 `query/pos/neg` 스키마를 검사하며, 기존 mined-negative 파일과 동등성을 주장하지 않는 새 선택입니다. 정확히 `train_minedHN.jsonl`이 필요한 실험은 `files`를 명시하고 경로가 없으면 실패하도록 유지합니다. unique basename 탐색은 `nested_explicit=true`를 명시했을 때만 허용합니다. 데이터 의미가 바뀐 실행은 새 baseline·source manifest로 기록해야 합니다.

기본 Auto5M 목표는 그대로 **500만 학습 판단**이며 검증 10,000 / 온도 보정 4,000 / 신규 테스트 14,000입니다. 미사용 conformal 파일은 길이 0인 호환 캐시로 남을 수 있지만 학습·보정에 사용하지 않습니다. 이미 사용한 보정 자료를 최종 테스트로 옮기지 않습니다. 계획의 1,100만 학습·7만 평가 물량을 이번 PR이 확보했다는 뜻은 아닙니다.

## 지시문 쌍 만들기와 학습

원본 JSONL의 **train 행**에만 검토한 `instruction_views`를 넣습니다. 변형할 수 있는 것은 질문·후보 설명·언어 메타데이터이며 원본 state·정답·partition을 덮어쓰는 항목은 거부합니다.

```json
{"id":"case-1","group_id":"origin-1","partition":"train","language":"ko","state":"코드를 수정했고 테스트는 실행하지 않았다.","type":"choice","question":"다음 작업은?","criteria":{"test":"테스트 실행","finish":"완료 처리"},"target":"test","instruction_views":[{"view_id":"reviewed-1","reviewed":true,"question":"어떤 일을 다음에 해야 하는가?","criteria":{"finish":"작업 종료","test":"변경 사항 검증"},"language":"ko"}]}
```

```bash
python -m nanodecision2.instruction_views --input data/reviewed.jsonl \
  --out data/instruction-pairs.jsonl --fraction 0.25 --seed 42
nd2 prepare --input data/instruction-pairs.jsonl --init init/base \
  --out data/cache-v12 --max-state 2048 --max-pair 256
nd2 train --init init/base --cache data/cache-v12 --out runs/v12 \
  --device cuda --bf16 --batch-size 4 --accumulation 16 \
  --workers 2 --prefetch 2 --pair-consistency 0.05 \
  --selection-metric macro_task_error
```

파일에는 별도 고정 validation/temperature/test 자료도 있어야 합니다. 위 한 줄 예제만으로 학습할 수는 없습니다. `init/base` 대신 감사된 기존 체크포인트를 사용하면 warm start입니다. 신규 코드는 학습/배포 그래프 변경 없이 기존 가중치를 읽습니다. `--pair-consistency`는 검토된 쌍이 실제로 있을 때만 켭니다. 기본 Auto5M이 임의 패러프레이즈를 자동으로 생성·승인하지는 않습니다.

두 번째 view는 독립 원본이 아닙니다. 결과 manifest에서 source_rows/output_rows/reviewed_pairs를 분리합니다. 같은 문서의 모든 질문·번역·변형은 같은 원본 그룹으로 유지해야 합니다. 같은 split 안에서도 중복/라벨 검사를 계속 수행합니다.

## 병렬화 검증 범위

읽기 전용 가중치와 immutable 컨텍스트를 공유하되 C++ 후보 캐시의 변경은 짧게 잠급니다. `clear_cache`와 동시에 계산하던 이전 세대의 결과는 캐시에 재삽입하지 않습니다. Python 토큰 캐시 역시 엔진별로 한도를 둡니다. CPU/BLAS 스레드 과잉은 사용 환경에서 별도로 조정해야 합니다.

`--workers`는 실제 GPU batch를 키우는 옵션이 아니라 압축 캐시 I/O를 겹치는 기능입니다. worker 수를 바꾸더라도 후보 셔플은 학습 프로세스의 RNG를 쓰고, 체크포인트는 소비한 batch offset을 저장합니다. CPU에서는 연속 학습과 workers=2 중단 후 workers=0 재개의 가중치 일치를 시험했습니다. CUDA bitwise 동일성·처리량 향상은 측정하지 않았습니다.

## 다음 단계 — 아직 구현/검증하지 않은 부분

| 계획 요구 | 다음 작업 |
| --- | --- |
| 병렬 수집·전처리 | 다운로드/토큰화 bounded queue 및 다중 source coordinator; 현재는 사전검사와 병렬 cache loader 중심 |
| 학습 GPU 최적화 | batch 128/256 probe, token-budget sampler, pinned transfer, compile, DDP/FSDP, 영어 과제별 보호 게이트 |
| 새로운 학습 감독 | 근거·관계 auxiliary heads, 프로그램 검증 데이터 대량 생성, R-Drop/Few-shot 소거실험 |
| 인코더·컨텍스트 | 교차 토큰 결합층, 4K/8K 추가학습 및 평가, 16K/32K 계층형 처리; 기존 길이 한도를 임의 확대하지 않음 |
| 구조화·센서 | 타입·단위·마스크 계약, 수치·시계열 인코더, 별도 학습/네이티브 형식·parity |
| 한영 혼합 | 8개 언어 조합·code-switch 학습셋 확보 및 전체 문항 평가; 현재는 metadata 경로·reviewed views 제공 |
| JevBench | 공식 버전/원 채점기를 고정한 local adapter, 공개 코호트·지원률·실패·비용/latency 구분 |
| 외부 모델 | Julia/Laya 실가중치 비교, 새 모델의 정확도·확률 개선 확인 |

## 실행한 검증

검증 결과와 명령은 [`VALIDATION_V1_2_FOUNDATION.md`](VALIDATION_V1_2_FOUNDATION.md)에 기록합니다. 테스트는 작은 CPU fixture와 모의 원천을 사용하며 모델 정확도·500만 행 확보·CUDA 처리량의 증거가 아닙니다. PR은 `main`에 자동 병합하지 않습니다.
