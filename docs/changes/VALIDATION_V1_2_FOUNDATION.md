# v1.2 1차 구현 검증 — 2026-11-13

이 문서는 이번 PR의 실행 기록입니다. 과거 0.3.1 기록을 새 테스트 결과로 재사용하지 않았습니다.

## 직접 실행

- Linux x86-64, Python 3.13.5, PyTorch 2.10.0+cpu; CUDA 없음.
- C++ 확장 재빌드 성공: `python setup.py build_ext --inplace`.
- 전체 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m pytest -q`: **138 passed in 13.40s**.
- 기존 테스트의 smoke 모델 사용은 `allow_smoke=True`를 명시하도록 변경하고, 새로운 기본 non-null 동작을 검사했습니다. 기존 그룹 정규화/누출 방지 변경과 테스트도 유지했습니다.
- 새 수용시험: 기본·레거시 출력, 낮은 확률, 기존 qhat, 미보정 모델, 동점·후보 순서, Noul 이름 매핑, Score 기댓값/최빈값, 파일 손상, CLI mixed 온도 전달.
- 수집 회귀: literal 따옴표, 실제 quoted 탭·개행, BOM·CRLF, gzip·record 재개, 빈 마지막 열, 필드 한도 복원, 잘못된 열 수·헤더, plain_text·중첩 파일·variant 누락·메타데이터·모호한 경로.
- 병렬 추론: 동일 C++ 모델에 8개 요청 스레드로 320회, 서로 다른 context와 cache hit/miss/clear를 섞어 순차 기준 logits와 비교했습니다. ThreadSanitizer나 실제 서비스 부하는 실행하지 않았습니다.
- 병렬 학습 입력: spawned worker 0/2에서 같은 row 순서, worker=2에서 중단 후 worker=0으로 재개한 CPU 학습과 연속 학습의 최종 가중치 bitwise 일치.
- 지시문 view: 검토되지 않은 변형/정답 주입/후보 변경/평가 자료 변형 거부, 후보 순서별 soft target 정렬, 같은 언어의 검토된 JS 쌍, 원본·view 수 분리.
- 과제별 macro accuracy 기반 모델 선택의 작은 학습 경로 확인.
- 소형 Auto 파이프라인: source fixture 단계 → 학습 → INT8 → temperature-only → 평가 → bundle 완료. train 120, validation 20, temperature 20, conformal 0, test 40.
- 해당 시험은 `reference_fallback_no_verified_mixed_improvement`로 기준 가중치를 보존했습니다. 개선을 확인한 실제 언어 모델이라는 뜻이 아닙니다.
- 종료된 실행을 동일 설정으로 다시 실행해 **15개 단계가 `resumed_skip`**하는 것을 확인했습니다.
- Python `compileall`, Bash `bash -n`, `git diff --check` 통과.
- wheel 빌드 및 별도 디렉터리 설치 성공. 작업 트리 밖에서 torch/NumPy/Transformers/datasets/pyarrow/ONNX Runtime import를 차단한 상태로 C++ fixture 추론 성공, API 계약 2·non-null 출력 확인.

## 실행하지 않은 것

네트워크 제한으로 실제 백본과 데이터 shard를 다운로드하지 못했습니다. tokenizers/transformers/pyarrow가 이 검증 환경에 없으므로 native 수치 시험에는 byte-token 소형 모델을 사용했습니다. live resolver는 모의 API/파일 목록과 알려진 원천 계약으로 시험했으며, 실제 원천의 현재 모든 파일·스키마를 검증한 결과는 아닙니다.

**500만/1,100만 판단 자료 확보, CUDA·BF16 본학습, RTX 5090 처리량·VRAM, 새 한국어/영어 정확도, 8K 이해력, 센서 인코더, JevBench·Julia·Laya 실가중치 비교는 미실행입니다.** 데이터 손상 모델의 재학습 여부와 source variant 변경 효과는 실제 깨끗한 자료에서 재평가해야 합니다.

테스트 통과는 코드 계약과 소형 경로의 검증입니다. 정확도·속도 향상 수치나 제품 출시 승인을 의미하지 않습니다.
