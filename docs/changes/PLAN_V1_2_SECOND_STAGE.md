# v1.2 후속 구현: 분산 학습·다중 입력·장문·JevBench

2026-11-13 / PR #3 후속. **코드 경로 구현과 모델 능력 향상은 별개다.**
기본 판단은 항상 최선의 후보와 확률을 반환한다. answerability 게이트나
자동 unknown/null을 새로 넣지 않는다. 원래 NLI의 neutral 같은 정답은 보존한다.

## 구현된 범위와 경계

| 영역 | 이번 코드 | 검증/제한 |
|---|---|---|
| 배치·DDP | 패딩 토큰 예산, 마지막 불균등 rank의 zero-weight 처리, no_sync 누적, 전역 가중 평균 gradient | 2-process CPU/gloo 시험. CUDA/NCCL·실제 큰 모델 속도는 미측정 |
| 배치 탐색 | forward/backward/AdamW 상태까지 사용하는 micro-batch probe | 실측한 장치/표본의 결과만 기록. 입력 자르기·자동 production 설정 변경 없음 |
| 병렬화 | 원천 첫 shard preflight 동시 다운로드, 제한된 queue, spawn 토큰화, 단일 audit writer | Auto5M Store 전체 변환을 다중 writer로 바꾸지는 않음. `prepare --workers`는 로컬 정제 JSONL에 적용 |
| 한영·지시문 | 8개 state/question/candidate 조합, 문장 내 mixed, 역할/조건 대조, oracle 라벨, 0/1/4-shot 자료 생성 | 합성 템플릿 생성기. 실제 한국어·영어 능력 향상을 아직 측정하지 않음 |
| 학습 손실 | NLL/Brier/ordinal + 선택 JS/R-Drop, 과제별 영어/한국어 회귀 검사 | 전체 문항 기준. 높은 확률만 보상하지 않음 |
| 표·레코드 | 타입/단위/0/null/부재 구분, 행 내 필드 관계를 결합하는 학습 모듈 | 선택 PyTorch 연구 backend. float feature만으로 정확한 큰 정수 연산을 보장하지 않음 |
| 원시 센서 | 값·관측 mask·시간차·채널/단위의 패치 encoder | 선택 PyTorch 연구 backend. 실제 기기/세션 데이터로 별도 학습 필요 |
| 토큰 결합·보조 감독 | 풀링 전 교차 비교, 질문 조건 근거 start/end·관계 보조 head | 감독 없으면 해당 loss만 생략. 근거 검출 실패로 답을 취소하지 않음 |
| 장문 | 직접 4K/8K 학습 프로필, C++ all-block key/value 결합(합계 최대 32K) | all-block은 위치를 블록마다 초기화하며 direct 32K가 아님. 길이별 품질 미인증 |
| JevBench | 실제 upstream Task/Runner/scoring/summarize와 연결하는 local adapter | 공개 코호트 전용. 공식 전체 점수/비공개 점수는 N/A. 실제 대형 가중치 점수 미측정 |

센서/표/새 token-interaction 그래프는 `nd2-extended-research-v1`로 저장한다.
기존 C++ 포맷으로 잘못 export하면 오류다. **신규 수치 모듈을 아직 torch 없는 C++
배포 경로에 구현한 것은 아니다.** 기존 텍스트 C++ 런타임은 그대로 사용한다.
FSDP, GPU 동적 요청 scheduler, 전체 Auto5M 병렬 ingest, 실데이터 대규모 학습은 남아 있다.

## 설치와 기존 모델 보존

기존 번들과 checkpoint는 덮어쓰지 않는다. 파서/코드/레시피가 바뀐 실행에는 새 workdir를 쓴다.
학습용 PyTorch는 장치에 맞는 공식 휠을 별도로 설치한다. 아래 명령은 저장소 루트에서 실행한다.

```bash
python -m pip install --no-build-isolation -e '.[train,test]'
python -m pytest -q
```

## 자동 파이프라인: 유효 배치 64 / 128 / 256

```bash
# 데이터 준비까지만: 실제 모델·공개 원천 다운로드가 발생한다.
bash run_all.sh --training-profile throughput64 --workdir "$HOME/nd2-runs/v12-scalable" --stop-after-stage data

# 이미 구성한 실제 캐시에서 학습 micro-batch 비용을 측정한다.
.venv031/bin/nd2 probe-batch --init "$HOME/nd2-runs/v12-scalable/init" \
  --cache "$HOME/nd2-runs/v12-scalable/cache" --device cuda --bf16 \
  --candidates 1 2 4 8 --out "$HOME/nd2-runs/v12-probe.json"

# 같은 설정으로 실행하면 완료된 단계 뒤에서 진행한다.
bash run_all.sh --training-profile throughput64 --workdir "$HOME/nd2-runs/v12-scalable"
```

`batch128`, `batch256`은 각각 새 workdir의 비교 실험이다. profile 기본 micro-batch가
장치에 맞는다는 보장은 없다. OOM이면 **명시적으로** custom recipe의 micro-batch를
줄이고 새 실행을 시작한다. 데이터 길이를 조용히 잘라 맞추지 않는다.

직접 장문 프로필은 `context4k` 또는 `context8k`다. 해당 길이의 유효 데이터,
encoder 위치 한도, VRAM, 짧은 문맥/영어 회귀를 함께 검증해야 한다.
기존의 짧은 데이터만으로 8K 정확도가 검증됐다고 표시하지 않는다.

기본 자동 학습에는 기존 영어 기준 단계와 한영 혼합 단계가 있다. `trainer=scalable`은
과제×언어 정확도/NLL/Brier를 각각 검사한다. 최종 양자화 후보 평가에도 이 검사를 적용한다.
CLI에서 `--no-language-guard`는 명시적인 연구용 소거실험이며 출시 승인이 아니다.

## 직접 분산 학습

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m nanodecision2.scalable_training --ddp \
  --init /path/to/checkpoint --cache /path/to/audited-cache --out runs/ddp-new \
  --device cuda --bf16 --batch-size 4 --accumulation 16 \
  --max-tokens 24000 --workers 2 --prefetch 2 --pin-memory \
  --protect-language en --protect-language ko
```

고정 크기일 때 유효 배치는 `4 × 16 × 2 = 128`이다. token budget으로 micro-batch가
작아지면 실제 update 행 수가 달라지며 전역 가중 합으로 정확히 정규화한다.
샘플을 복제해 마지막 rank를 채우지 않는다. 필요한 zero-weight 계산은 노출 수에서 제외한다.
`--rdrop 0.05`와 `--pair-consistency 0.05`는 별도 실험이다. JS에는 검증된 2개 view가 필요하다.

`--resume runs/ddp-new/last.pt`는 같은 world size·데이터·코드·학습 설정에서만 허용한다.
workers 변경은 허용한다. 정기 checkpoint 이후의 미저장 작업은 재실행될 수 있다.
이 trainer는 모든 rank의 OOM을 자동 복구하거나 임의 Ctrl+C 시 즉시 저장한다고 약속하지 않는다.

## 검증 가능한 합성 데이터와 병렬 전처리

```bash
nd2 generate-verified --worlds 10000 --seed 42 --fewshot 0 --out data/verified.jsonl
nd2 prepare --input data/verified.jsonl --init /path/to/init --out data/verified-cache \
  --max-state 2048 --max-pair 512 --workers 4 --prefetch 2
```

1 world에서 5개 과제의 60개 표현을 만든다. 10,000 world는 600,000 판단 행이지만
독립 원본 600,000개가 아니다. 원본 그룹별 split은 먼저 고정되며 동등 view는 같은 split이다.
역할·조건 정답은 프로그램으로 계산한다. 템플릿 기반 학습용 자료이며 실제 센서 고장 데이터가 아니다.
`--fewshot 1`/`4`는 별도 원본의 support 예시를 붙인다. query 정답은 input에 넣지 않는다.
실데이터와 혼합하고, 합성 템플릿의 성적만으로 일반화 성능을 주장하지 않아야 한다.

병렬 토큰화는 bounded queue를 쓰며 최종 그룹 감사/캐시 쓰기는 coordinator 한 개가 수행한다.
worker가 처리한 순서가 아니라 원본 순서로 게시한다. 기존 JSONL과 원본을 수정하지 않는다.
실패한 출력 폴더는 완료 cache가 아니며 새 출력 폴더에서 재시도한다.

## 구조화·센서 연구 모듈의 학습과 실행

학습 JSONL은 `state`, `question_spec`, `supervision`을 분리한다.
모든 행에 `id`, `group_id`, `partition=train|validation|temperature|test`를 명시한다.
사람·기기·세션을 먼저 나눈 뒤 window를 만들고, 같은 그룹을 여러 split에 놓지 않는다.

```json
{"id":"train-001","group_id":"device-session-001","partition":"train","language":"en","task":"threshold","state":{"modality":"records","fields":[{"name":"temperature","unit":"C","value":35}]},"question_spec":{"type":"noul","question":"Is temperature at least 30 C?"},"supervision":{"target":true}}
```

서로 다른 그룹의 validation/temperature 자료도 준비한 뒤:

```bash
nd2 train-extended --init /path/to/checkpoint --input data/multimodal.jsonl \
  --out runs/extended-new --structured --sensors --token-interaction \
  --block-tokens 2048 --max-context 32768 --device cuda --gradient-checkpointing \
  --protect-language en --protect-language ko
nd2 extended calibrate --bundle runs/extended-new/best --input data/multimodal.jsonl
nd2 extended predict --bundle runs/extended-new/best --input examples/sensor_request.json
```

새 모듈은 실제로 학습해야 한다. label 없이 켜기만 하면 성능이 생기지 않는다.
이 연구 trainer는 현재 단일 장치, FP32, row-wise accumulation이며 대규모 고속 DDP는 text trainer와 구분한다.
연속 회귀는 지원하지 않고 Choice/Noul/Score를 사용한다. 센서 데이터는 inline `values[T,C]`,
`observed_mask`, `channel_names`, `units`, `timestamps` 또는 `sample_rate_hz`를 사용한다.
결측은 mask와 JSON `null`로 표현한다. 임의 파일/URL 참조는 열지 않는다.
`evidence_token_span`은 STATE prefix를 포함한 실제 토큰 offset이다. 문자 offset을 그대로 넣으면 안 된다.

## C++ 장문 all-block 실행

```bash
nd2 predict-long --bundle /path/to/native-bundle --input request.json \
  --block-tokens 2048 --max-tokens 32768
```

기존 번들의 직접 입력 한도보다 block size를 크게 지정할 수 없다. 모든 block을 읽고
key/value 특징을 합친 후 기존 판단 head가 전체를 참조한다. 블록별 확률을 평균하지 않는다.
토큰 offset과 실제 처리량을 반환한다. 기존 direct calibration은 자동 재사용하지 않는다.
별도로 검증한 `all_blocks_calibration.json`이 없으면 raw probabilities라고 표시한다.
문맥 간 결합이 부족할 수 있으므로 실제 장문 학습/평가는 여전히 필요하다.

## JevBench 공개 평가

별도 JevBench checkout을 설치하고 commit을 고정한다. 검토한 계약의 기준 commit은
`609a664e81f46e03731600bd7bd9f1b7a61257c5`이며, 미래 버전과 자동 호환을 보장하지 않는다.

```bash
nd2 jevbench --bundle /path/to/native-bundle --tasks /path/to/public-tasks.jsonl \
  --upstream-commit 609a664e81f46e03731600bd7bd9f1b7a61257c5 --out /path/outside/jevbench/run
```

공식 `Runner`, `score_task`, `summarize`를 사용한다. gold는 모델 입력에 전달하지 않는다.
Noul no/yes ↔ false/true는 이름으로 대응하고, Score의 모든 레벨을 보존한다.
public-only 점수이며 비공개 composite/공식 순위는 N/A다. 로컬 비용은 미측정 null이다.
실패는 원래 harness 정책에 남기며, 중단 시 미시도 수와 전체 예정 문항의 성공 비율을 별도로 기록한다.
모델·토크나이저·task·scorer·runner hash, 환경, raw response, native logits를 저장한다.
Jev API 키는 사용하지 않는다. 공개 평가 문제는 학습/보정에 포함하지 않는다.

## 남아 있는 검증/구현

실제 5M/11M 수집·학습, CUDA/NCCL·BF16 처리량, 4K/8K 정확도,
실기기 센서/표의 held-out 성능, Julia/Laya/JevBench 실가중치 비교는 수행하지 않았다.
새 수치/결합 그래프의 C++ export, FSDP, 전체 Auto5M parallel ingest, GPU 요청 동적 배치는 미완료다.
코드가 정상 실행된 것과 목표 품질·속도를 달성한 것을 혼동하지 않는다.
