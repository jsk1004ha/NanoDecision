# 본문과 발화를 함께 읽는 대규모 후보 선택

계정이나 서비스가 본문에 있고 사용자가 원하는 동작이 발화에 있다면 두 필드를 함께 모델에 전달합니다. 예를 들어 똑같이 “취소해 줘”라고 말해도 현재 화면이 항공 예약인지 상품 주문인지에 따라 정답이 달라집니다. 아래 경로는 학습과 추론 양쪽에서 `{"context": 본문, "utterance": 발화}`를 모델 상태로 사용합니다.

## 학습 자료 구성

| 자료 | 질문당 후보 수 | 학습 신호 |
| --- | --- | --- |
| Bitext의 공개 영어 학습 자료 | 100, 200, 300 또는 전체 | 발행자가 붙인 의도와 실제 입력에 노출된 서비스 도메인을 결합합니다. 동일한 의도명이 여러 도메인에 있으면 도메인별 후보가 생깁니다. |
| 직접 작성한 한국어·영어 합성 자료 | 32, 100, 200, 300, 1,000 | 서비스 계정 10개 × 제작물 10종 × 작업 10종입니다. 계정만 바꾸거나 요청만 바꾸면 정답이 바뀌고, 화면 테마처럼 무관한 필드만 바꾸면 정답을 유지합니다. |

```bash
nd2 prepare-bitext --download --out data/bitext-300 --candidates 300 --max-per-intent 20
nd2 generate-grounded --out data/grounded-1000.jsonl --k 1000 --language both --limit-groups 2
```

첫 명령은 지정한 Bitext 학습 저장소 11개의 리비전·파일 크기·SHA-256을 검사합니다. 고정된 후보 목록을 모든 행에 사용하며, 같은 도메인과 정규화된 발화에서 서로 다른 정답이 관측되면 해당 입력 전체를 제외합니다. `--source-dir`로 검증할 로컬 CSV 폴더를 지정할 수 있습니다. `--max-per-intent 0`은 선택된 의도의 모든 적격 행을 출력하므로 캐시 디스크 용량을 먼저 확인하세요. 원천과 제외 건수는 `manifest.json`에, 후보 설명은 `catalog.json`에 기록합니다. 생성 자료 배포 시 출력 폴더의 `NOTICE.md`와 `LICENSE-DATA.txt`를 함께 유지하세요. [고정된 원천과 라이선스](../nanodecision2/recipes/bitext_training_sources.json)

고정 원천에서 기존 평가 발화와 겹친 것으로 확인한 학습 발화 지문 8개는 기본 제외 대상입니다. 도메인 간 중복을 포함해 정제 후 학습 행 9개에 해당합니다. 별도로 보유한 평가 발화와의 정확 일치도 제거하려면 `--heldout-fingerprints hashes.json`을 추가하세요. 파일은 발화를 NFKC·casefold 처리한 뒤 유니코드 문자·숫자 단어를 공백 하나로 이어 계산한 SHA-256 문자열의 JSON 배열입니다. 이 필터의 범위는 지문과 정규화 후 일치하는 발화입니다. 번역, 바꿔 쓴 문장, 공유 문서는 그룹 단위로 따로 검사해야 합니다. 두 명령 모두 학습 행만 생성하며 JevBench와 Decision Index 문항은 읽지 않습니다.

`--max-per-intent`로 개수를 제한하면 각 의도에서 정규화 발화 정렬 순서로 앞선 예시를 출력합니다. 이 작은 표본으로 전체 말투나 표현의 분포를 추정하기는 어렵습니다.

둘째 명령은 두 계정과 두 요청을 교차한 그룹 2개를 만듭니다. 각 그룹의 네 가지 조합 및 무관한 본문 변경 예시는 동일한 후보와 표시 순서를 공유합니다. `--limit-groups`를 생략하면 250그룹을 생성합니다. 후보 1,000개는 계정별 운영 경로 수이며, 독립적인 자연어 의도의 다양성은 별도 측정 대상입니다.

## 준비와 학습

기존 학습 초기화 디렉터리를 `init/model`에 두고, 새로 작성하거나 학습 전용 자료에서 별도로 떼어 둔 검증 그룹을 `data/independent-validation.jsonl`에 준비합니다. 검증 행은 `partition="validation"`으로 명시하고 같은 본문, 발화 변형, 번역 그룹은 한 분할 안에 유지하세요. 아래 명령은 두 자료와 검증 파일을 실제 학습 캐시로 변환합니다.

```bash
nd2 prepare --init init/model --input data/bitext-300/train.jsonl.gz data/grounded-1000.jsonl data/independent-validation.jsonl --out cache/grounded --max-state 2048 --max-pair 256
nd2 train --init init/model --cache cache/grounded --out runs/grounded --device cuda --batch-size 1
```

직접 선택은 후보 전체에 대해 하나의 경쟁 점수와 확률 분포를 계산합니다. 넓은 입력은 후보별 인코딩을 나누어 계산하고 학습 중 중간값을 재계산합니다. 후보 개수와 토큰 길이가 함께 계산 비용에 영향을 주므로 실제 모델과 하드웨어에서 배치 크기를 확인하세요.

## 추론

후보 2–1,000개에서 전체 확률이 필요하면 기존 `nd2 predict` 또는 `DecisionEngine.decide`에 모든 후보를 전달합니다.

```python
from nanodecision2 import DecisionEngine

engine = DecisionEngine("bundles/trained")
state = {"service_domain": "travel"}
utterance = "취소해 줘"
criteria = {"air_cancel": "항공 예약 취소", "shop_cancel": "상품 주문 취소"}
question = "현재 서비스와 요청에 맞는 동작을 선택하세요."
result = engine.decide(
    {"context": state, "utterance": utterance},
    {"route": {"type": "choice", "question": question, "criteria": criteria}},
    language="ko",
)
```

큰 목록에서 계산할 후보 수를 줄이려면 `request.json`의 루트에 `state`, `utterance`, `question`, `criteria`와 선택적인 `language`를 넣습니다. Bitext로 만든 `catalog.json`은 후보 ID를 설명에 대응시키는 JSON 객체입니다. 이 객체를 그대로 `criteria`에 넣습니다.

```bash
nd2 predict-catalog --bundle bundles/trained --input request.json --top-k 32
```

검색은 본문·발화와 후보 설명의 문자 및 단어 겹침을 사용합니다. 검색된 후보 중 최대 256개를 모델이 다시 평가합니다. 응답의 `retrieval.shortlist`에서 후보와 순위를 확인할 수 있습니다. 전체 후보 중 일부만 통과했다면 `probability_scope`는 `conditional_on_retrieved_shortlist`입니다. 이 확률은 검색 단계에서 정답을 놓칠 가능성까지 반영하지 않습니다. 기존 번들의 후보 수별 보정 표는 21개 이상을 한 구간으로 묶으므로, 100개와 1,000개에서의 확률 신뢰도는 각각 검증해야 합니다.

일부 후보만 검색한 경우 응답은 검색을 포함한 전체 경로의 `calibrated`를 `false`로 표시합니다. `temperature_calibrated`는 번들의 온도 보정 여부를 따로 보여 줍니다. `--min-confidence`를 양수로 지정하면 검색에서 빠진 정답의 확률을 알 수 없어 검토 사유가 추가됩니다. 이 상황에서 `--legacy-policy`를 켜면 `value`가 보류되고 `selected_option_id`로 모델의 1위 후보를 확인할 수 있습니다.

## 개선 효과 측정

같은 독립 평가 그룹에서 기존 가중치와 새 가중치를 비교합니다. 각 후보 수에서 전체 선택 정확도와 NLL을 기록하고, 두 단계 추론에서는 정답의 검색 통과율(Recall@K), 통과한 경우의 선택 정확도, 전체 요청의 최종 정확도를 함께 보고하세요. 본문만 바꾼 쌍과 발화만 바꾼 쌍은 각각 두 행을 모두 맞힌 비율로 집계합니다. 무관한 본문 변경 및 후보 순서 변경은 동일한 정답을 유지한 비율로 집계합니다. 영어와 한국어는 따로 보고하여 한 언어에서의 퇴행을 확인할 수 있습니다.
