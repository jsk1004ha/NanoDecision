# Julia / Laya / NanoDecision2 공통 비교 규약 — 0.3.0

## 미리 고정할 것

원본 데이터 revision/hash, 입력 state, 질문, 모든 후보 ID/설명/순서, gold 의미,
언어, 모델 파일/SDK 버전, 토큰 제한, device/정밀도/스레드, 보정의 학습 범위를 고정합니다.
테스트 결과를 보고 prompt, temperature, threshold를 바꾸지 않습니다.

### 트랙

A. 한국어: MASSIVE-ko 18 scenario 전체 후보, PAWS-X-ko, KLUE-NLI 공개 dev(개발 성적으로 표시),
새로 독립 수집한 한국어 업무·로그·조건 문항을 과제별로 보고합니다.
공식 공개 dev를 반복 사용한 뒤 test라고 이름만 바꾸지 않습니다.

B. 영어 유지: 같은 유형의 English 공식 평가로 한국어 개선이 영어 붕괴를 유발하지 않는지 확인합니다.
병렬문장은 같은 source group으로 bootstrap합니다.

C. typed benchmark: teacher-label agreement입니다. Laya typed specialist, Laya multilingual/English,
Julia, ND2를 각각 명시하며 specialist 여부와 typed train 노출을 공개합니다.
동일 gold.label 정확도와 teacher-probability NLL/Brier를 별개로 보고합니다.
Score는 argmax option으로 accuracy, 기대 점수로 별도 error를 구합니다.
`gold.label`이 probability mode와 다를 수 있으므로 둘을 바꾸지 않습니다.

D. 용량/배포: 전체 60/77후보, 긴 상태, 다중 질문, CPU precision 같은 조건을 따로 검사합니다.
공통 <=20 native 비교와 이 대용량 트랙을 합쳐 편리한 점수 하나로 만들지 않습니다.
현재 raw runner는 한 행 한 질문 latency를 측정합니다. 여러 질문 context-sharing 처리량은 이 버전의 별도 실측으로 제공하지 않았습니다.

## 실행

각 환경에서 `scripts/evaluate_raw.py` 실행 후 3개의 JSONL을 모읍니다.
`--engine laya --laya-variant multilingual --model LOCAL_DIRECTORY`가 한국어 기본 대조군입니다.
Laya의 자동 Router가 입력에 따라 다른 checkpoint를 몰래 선택하게 하지 않습니다.

baseline 확률은 as-shipped, ND2는 실제 배포 후 별도 보정한 것을 평가하는 제품 비교를 기본으로 합니다.
이것은 학습 예산/보정 예산이 같은 연구 실험과 다릅니다. 동일한 보정 예산 실험을 할 때는
모든 모델의 raw logits를 별도로 얻어 같은 calibration-only groups에서 온도를 맞춰야 합니다.
현재 Julia named API의 shipped probabilities를 억지 역변환해 raw logits라고 간주하지 않습니다.

Laya 어댑터는 upstream `build_sequence`와 전체 렌더링을 비교하여 묵시적 잘림을 거부합니다.
따라서 긴 한국어 설명이나 많은 후보가 실제로 들어가지 않으면 status=failed로 기록됩니다.
Laya kwargs가 고정 검토 commit과 달라지면 실행을 검토하고 API 계약을 갱신하세요. 실패를 지원한다고 위장하지 않습니다.

## 지표 해석

- NLL, Brier(sum over options), target-mode agreement, reference-label accuracy, soft accuracy,
  ordered CRPS/score-mean error를 따로 기록합니다. Brier를 후보 수로 나누는 다른 구현과 숫자를 직접 비교하지 마세요.
- 고정된 정답 확률이 teacher 분포이면 NLL/Brier는 teacher agreement이지 사람 정답성 증거가 아닙니다.
- 입력 hash가 같지 않거나 요청 ID 집합이 다르면 비교를 거부합니다. 실패 행도 반드시 있어야 합니다.
- 세 모델 공통 지원 교집합의 품질과 각 모델의 전체 지원률/실패를 모두 보고합니다.
  실패가 많은 모델이 쉬운 문항만 풀어 높은 점수를 얻는 효과를 숨기지 않습니다.
- bootstrap은 source group을 재표집하고 행 가중 통계량을 보존합니다. 기본 2,000회이며 근사 구간입니다.
  전체 차이와 한국어별 차이를 기록합니다. 무조건적인 유의성 보장이나 다중 비교 보정은 아닙니다.
- latency는 import/load/hash/warmup/file IO 제외, tokenization/strict preflight/compute/decoding 포함입니다.
  모델마다 다른 내부 형식은 모델 구현의 일부이며 같은 숫자 token 제한이 같은 원문 양을 뜻하지 않습니다.
  실제 모두 입력을 읽었는지를 확인한 경우만 common-supported로 들어갑니다.
- RSS는 OS high-water mark이며 모델 로딩/파일 해싱/예측 결과 저장도 포함합니다.
  Python 유무를 제외하고 tensor 크기만 메모리 비교라고 제시하지 않습니다.

## Julia와 Laya를 넘었다고 주장하려면

새 독립 한국어 과제에서, 사전에 정한 핵심 정확도/확률 품질을 두 baseline과 각각 비교해야 합니다.
한 전체 평균에서만 이기고 한국어 조건·부정 과제에서 지는 것을 보편적 우월성이라 하지 않습니다.
원문/질문 길이와 실패를 함께 보고하고, 영어 유지 조건 및 메모리/속도 요구를 미리 정합니다.
단위 테스트 통과나 작은 synthetic smoke 성적은 이 조건을 대체하지 않습니다.
