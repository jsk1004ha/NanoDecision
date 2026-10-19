# Auto5M 기본 레시피 원천 목록

수량은 수집 상한이지 실제 정제 후 확보 수량이 아닙니다. 변환/제외 후의 최종 집계는 data_report.json을 확인하세요.

| 원천 ID | 공개 저장소 / 전송 | 구성·언어 | 분할 | 수집 판단 상한 | 필수 |
|---|---|---|---|---:|---|
| paws-en-train | google-research-datasets/paws-x | en / en | train | 60000 | 예 |
| paws-ko-train | google-research-datasets/paws-x | ko / ko | train | 60000 | 예 |
| paws-de-train | google-research-datasets/paws-x | de / de | train | 60000 | 예 |
| paws-es-train | google-research-datasets/paws-x | es / es | train | 60000 | 예 |
| paws-fr-train | google-research-datasets/paws-x | fr / fr | train | 60000 | 예 |
| paws-ja-train | google-research-datasets/paws-x | ja / ja | train | 60000 | 예 |
| paws-zh-train | google-research-datasets/paws-x | zh / zh | train | 60000 | 예 |
| massive-en-US-train | massive | - / en-US | train | 공개 분할 끝까지 | 예 |
| massive-ko-KR-train | massive | - / ko-KR | train | 공개 분할 끝까지 | 예 |
| massive-de-DE-train | massive | - / de-DE | train | 공개 분할 끝까지 | 예 |
| massive-es-ES-train | massive | - / es-ES | train | 공개 분할 끝까지 | 예 |
| massive-fr-FR-train | massive | - / fr-FR | train | 공개 분할 끝까지 | 예 |
| massive-ja-JP-train | massive | - / ja-JP | train | 공개 분할 끝까지 | 예 |
| massive-zh-CN-train | massive | - / zh-CN | train | 공개 분할 끝까지 | 예 |
| paws-en-test | google-research-datasets/paws-x | en / en | test | 공개 분할 끝까지 | 예 |
| paws-ko-test | google-research-datasets/paws-x | ko / ko | test | 공개 분할 끝까지 | 예 |
| paws-de-test | google-research-datasets/paws-x | de / de | test | 공개 분할 끝까지 | 예 |
| paws-es-test | google-research-datasets/paws-x | es / es | test | 공개 분할 끝까지 | 예 |
| paws-fr-test | google-research-datasets/paws-x | fr / fr | test | 공개 분할 끝까지 | 예 |
| paws-ja-test | google-research-datasets/paws-x | ja / ja | test | 공개 분할 끝까지 | 예 |
| paws-zh-test | google-research-datasets/paws-x | zh / zh | test | 공개 분할 끝까지 | 예 |
| massive-en-US-test | massive | - / en-US | test | 공개 분할 끝까지 | 예 |
| massive-ko-KR-test | massive | - / ko-KR | test | 공개 분할 끝까지 | 예 |
| massive-de-DE-test | massive | - / de-DE | test | 공개 분할 끝까지 | 예 |
| massive-es-ES-test | massive | - / es-ES | test | 공개 분할 끝까지 | 예 |
| massive-fr-FR-test | massive | - / fr-FR | test | 공개 분할 끝까지 | 예 |
| massive-ja-JP-test | massive | - / ja-JP | test | 공개 분할 끝까지 | 예 |
| massive-zh-CN-test | massive | - / zh-CN | test | 공개 분할 끝까지 | 예 |
| ag-news-train | fancyzhx/ag_news | - / en | train | 140000 | 예 |
| nsmc-train | e9t/nsmc | - / ko | train | 160000 | 예 |
| ag-news-test | fancyzhx/ag_news | - / en | test | 공개 분할 끝까지 | 예 |
| nsmc-test | e9t/nsmc | - / ko | test | 공개 분할 끝까지 | 예 |
| klue-nli-train | klue/klue | nli / ko | train | 공개 분할 끝까지 | 예 |
| klue-nli-validation | klue/klue | nli / ko | validation | 공개 분할 끝까지 | 예 |
| klue-ynat-train | klue/klue | ynat / ko | train | 공개 분할 끝까지 | 예 |
| klue-ynat-validation | klue/klue | ynat / ko | validation | 공개 분할 끝까지 | 예 |
| toolace | Team-ACE/ToolACE | - / en | train | 50000 | 선택 |
| enron | SetFit/enron_spam | - / en | train | 50000 | 선택 |
| support | Tobi-Bueck/customer-support-tickets | - / en | train | 40000 | 선택 |
| phishing | zefang-liu/phishing-email-dataset | - / en | train | 40000 | 선택 |
| yelp-score | Yelp/yelp_review_full | yelp_review_full / en | train | 450000 | 예 |
| snli | stanfordnlp/snli | - / en | train | 600000 | 예 |
| mnli | nyu-mll/multi_nli | - / en | train | 600000 | 예 |
| kornli-snli | https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/snli_1.0_train.ko.tsv | snli / ko | train | 600000 | 예 |
| kornli-mnli | https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/multinli.train.ko.tsv | multi_nli / ko | train | 600000 | 예 |
| ko-wiki-pairs | upskyy/ko-wiki-reranking | - / ko | train | 600000 | 예 |
| xnli-fr | facebook/xnli | fr / fr | train | 250000 | 예 |
| xnli-de | facebook/xnli | de / de | train | 250000 | 예 |
| xnli-zh | facebook/xnli | zh / zh | train | 250000 | 예 |
| kornli-snli-entails | https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/snli_1.0_train.ko.tsv | snli / ko | train | 600000 | 예 |
| kornli-mnli-entails | https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/multinli.train.ko.tsv | multi_nli / ko | train | 600000 | 예 |
| msmarco-evidence | microsoft/ms_marco | v2.1 / en | train | 2300000 | 예 |
