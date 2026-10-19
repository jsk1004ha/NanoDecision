# Laya 원천 검토와 한국어 데이터 설계

확인 기준: 2026-09-29. 나무위키 본문은 접근 제한으로 확인하지 못했습니다.
아래 기술적 근거는 Laya 제작자의 모델 카드·저장소·벤치마크 코드와 각 데이터 제작자의 자료입니다.

## 1. Laya에서 확인한 학습 노출: 전체 배합표는 아님

고정 코드: https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/research/scripts/bench_apps.py

| 원천 | 코드/카드에서 확인한 역할 | 이 버전 처리 |
|---|---|---|
| fancyzhx/ag_news | `in_training=True` | 4종 뉴스 분류 |
| SetFit/enron_spam | `in_training=True` | 스팸 Boolean |
| zefang-liu/phishing-email-dataset | `in_training=True` | Safe/Phishing Boolean |
| Tobi-Bueck/customer-support-tickets | `in_training=True`; 영어 10개 queue 선택 | queue 라우팅; 한국어 원천으로 간주하지 않음 |
| microsoft/ms_marco v1.1 | 학습 노출로 표시된 근거 선택 응용 | passage selection 약한 라벨. 미선택을 확정적인 무관련으로 바꾸지 않음 |
| LocalLLaMA/typed-decisions | 별도 typed specialist의 미세조정 자료 | 교사 확률·별도 reference label·원본 group 보존 |
| MASSIVE/XNLI | 다국어 평가에 사용 | 평가에 등장했다는 이유만으로 기본모델 학습 자료라고 단정하지 않음 |
| banking77, DAIR emotion, toxic-chat | 해당 응용 코드에서 `in_training=False` | 학습 노출이 있다고 잘못 분류하지 않음 |

이 표는 공개된 학습 노출의 일부 근거입니다. Laya 기본 체크포인트의 모든 데이터·샘플 수·배합 비율·정확한 훈련 ID를 복구한 표가 아닙니다.
특히 support/phishing의 일부 응용 평가가 원본 `train`에서 뽑혀 있다는 점 때문에 공개 응용 점수를 무조건 독립 test 점수로 간주해서는 안 됩니다.
공개 문구의 'real data'를 모든 행이 사람 원문/사람 정답이라는 뜻으로 확대하지 않습니다.

## 2. Typed decisions의 정답 의미

https://huggingface.co/datasets/LocalLLaMA/typed-decisions

train 1,200개 case / 6,000개 질문, test 400개 case / 2,000개 질문입니다.
공개 gold probability는 교사 모델의 반복 출력으로 만든 분포입니다. 이는 인간 독립 정답의 확률이 아닙니다.
latent facts는 입력에 넣지 않습니다. `gold.label`과 `gold.probabilities`의 argmax를 별도로 보존합니다.

자료를 사용하는 것 자체에 Jev API가 필요하지는 않지만, teacher-free 인간 정답 실험과는 별도 트랙이어야 합니다.
모델의 온라인 teacher API는 호출하지 않습니다. typed train은 선택 사항이며 기본 한국어 우월성 근거로 삼지 않습니다.
Conformal calibration에는 teacher soft labels를 넣지 않도록 변환·보정 단계에서 제한합니다.

## 3. 한국어 출발 자료의 실제 규모

| 원천 | 원본 학습 행 수 | 해석 |
|---|---:|---|
| KorNLI / SNLI 번역 | 550,152 | 영어 자료의 기계 번역 학습셋 |
| KorNLI / MultiNLI 번역 | 392,702 | 영어 자료의 기계 번역 학습셋 |
| KLUE-NLI | 24,998 | 한국어 NLI, neutral 보존 |
| KLUE-YNAT | 45,678 | 한국어 제목 주제 분류 |
| PAWS-X / ko | 49,401 | 한국어 train은 기계 번역; 평가 번역은 구분 |
| MASSIVE / ko-KR | 11,514 | 한국어 원본 locale의 train 발화; 18상황 또는 60의도 |
| 합계 | 1,074,445 | 원본 train 행의 단순 합계. 정제 후 유효 학습 행 수가 아님 |

KorNLI 두 자료만 942,854행이지만 독립된 한국어 원문 942,854개라는 뜻은 아닙니다.
PAWS-X 한국어 dev/test는 각각 2,000개, MASSIVE dev/test는 2,033/2,974개입니다.
KLUE의 공개 validation을 '비공개 공식 test'라고 부르지 않습니다.
MASSIVE 1.1의 100만+ 발화는 52개 locale 전체 합계이며 한국어가 100만이라는 뜻이 아닙니다.

원본 번역 관계·중복·보정/평가 분할을 제거하고 나면 학습량은 줄어듭니다.
변환기는 원본 train 내부를 group hash로 85/5/5/5(train/validation/temperature/conformal) 분할합니다.
그 비율 역시 실제 그룹 크기 때문에 정확히 맞지 않을 수 있습니다. 각 manifest가 실제 수량의 근거입니다.

## 4. 한국어 성능을 위한 구성 원칙

1. KorNLI 수가 많다는 이유만으로 실제 목적의 대부분을 NLI가 차지하게 두지 않습니다.
   영어 유지 자료, 한국어 원문 판단, 완료/실패 로그, 조건·예외·정보부족 사례를 구분한 실험을 합니다.
2. 한국어 original 문서의 추가학습은 선택 사항입니다. 사전학습은 번역 요청이나 API 증류가 아닙니다.
   원문을 tokenizer 기본 어휘로 학습하므로 구조·네이티브 커널을 바꿀 필요가 없습니다.
3. 조사/부정/시제/범위/숫자/단위/띄어쓰기를 삭제하는 정규화는 적용하지 않습니다.
   NFC 중복 감사는 자모 표현을 비교할 때 사용하지만 의미 단어를 지우지 않습니다.
4. 질문과 선택지도 한국어로 평가하는 트랙과, 원문 한국어 + 영어 schema 트랙을 분리합니다.
   MASSIVE scenario 설명 18개는 한국어로 작성했지만 별도 사용자 검토를 권합니다.
5. 한·영 대응 일관성은 같은 gold class만 가진 임의의 문장끼리 묶는 기능이 아닙니다.
   원본·질문·후보의 의미가 대응하는 두 행만 `equivalence_verified`를 승인합니다.
6. `language_weights`는 sampling을 자동으로 60:30:10으로 만드는 옵션이 아닙니다.
   보고서의 raw exposure/effective weight와 언어별 held-out NLL을 같이 확인합니다.

## 5. 한국어 비교에서 공개 점수를 섞지 않기

Laya multilingual 카드: MASSIVE **20후보 intent** 한국어에서 English checkpoint 0.110 → multilingual 0.490.
Julia 2026-09-24 raw metrics: MASSIVE **18 scenario** 한국어 2,296/2,974=77.20%, 영어 2,580/2,974=86.75%.
이는 서로 다른 과제입니다. 77.20와 49.0을 빼서 Julia가 Laya보다 그만큼 좋다고 판단하지 않습니다.
Laya 20후보 harness는 정답+19개 무작위 오답을 구성하므로 전체 60개 의도 분류와도 다릅니다.
새 비교는 먼저 동일한 18상황 전체 후보/동일한 한국어 test/동일한 설명으로 수행합니다.

Laya typed specialist의 76.6%와 기본 multilingual 성능도 섞지 않습니다.
Julia의 역사적 typed 73.15%는 H200 BF16이며, 2026-09-26 CPU FP32 재현의 426+542+483=1,451/2,000=72.55%와 구분합니다.
이 문서의 수치는 출처에서 공개한 조건부 결과이지 이번에 우리가 실행한 점수가 아닙니다.

## 출처

- https://huggingface.co/convaiinnovations/laya-multilingual
- https://huggingface.co/convaiinnovations/laya-typed-decisions
- https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/research/scripts/bench_local.py
- https://huggingface.co/SupersonicLabs/Julia-1
- https://huggingface.co/SupersonicLabs/Julia-1/blob/main/metrics/accuracy-20260924.json
- https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- https://huggingface.co/datasets/kakaobrain/kor_nli
- https://arxiv.org/abs/2004.03289
- https://huggingface.co/datasets/klue/klue
- https://huggingface.co/datasets/google-research-datasets/paws-x
- https://huggingface.co/datasets/AmazonScience/massive
- https://github.com/alexa/massive
