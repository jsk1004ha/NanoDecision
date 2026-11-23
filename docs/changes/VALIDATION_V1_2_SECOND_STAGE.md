# v1.2 후속 구현 검증 — 2026-11-13

이 기록은 PR #3의 첫 단계 기록과 구분한 후속 코드 검증이다. 소형 byte-token fixture의 학습·예측 성공은 실제 언어/센서 능력이나 처리량 개선의 증거가 아니다.

## 실제 실행

| 항목 | 결과 |
|---|---|
| 전체 회귀·신규 테스트 | **161 passed in 19.68s**. 첫 단계 138개에 신규 23개를 더한 실행 |
| 기존 PR #3 수용시험 | 기존 37개 모두 포함해 통과 |
| C++ 확장 | context_blocks.hpp / prepare_many 포함 재빌드, PyTorch 명시적 블록 특징 결합과 logit 대응 |
| DDP 전역 가중 손실 | 2개 CPU/gloo 프로세스, 서로 다른 표본 수와 가중치, no_sync 누적 후 단일 전역 기준 gradient 오차 ≤ 1e-12 |
| 실제 소형 분산 학습 | 13행, 마지막 불균등 rank, 총 노출 13행. 별도로 영어 6행만 선택한 2-process 학습도 완료 |
| CPU 중단·재개 | R-Drop 포함 같은 설정의 연속 학습과 재개 최종 가중치 bitwise 일치 |
| 병렬 토큰화 | workers 0/2의 출력 파일 hash와 cache manifest 일치 |
| 병렬 원천 preflight | 동일 첫 shard 경합을 직렬화하고 결과의 원천 순서 유지. 로컬 모사 시험이며 인터넷 처리량 측정이 아님 |
| 센서·표 | mask/시간/단위/shape, 0/null/부재, 열·행 순열 및 행별 필드 관계, 모듈 gradient 검사 |
| 연구 모듈 | 소형 센서 학습 → safetensors 저장 → 독립 temperature 보정 → 재로드·판단 → 파일 변조 거부 |
| 합성 데이터 | 8개 언어 조합, code-switch, 60표현/world, 정답 oracle·분할·0/1/4-shot query 정답 비노출 시험 |
| JevBench | 실제 upstream API 계약을 참조한 fixture에서 Noul/Score 매핑·확률 보존·gold 접근 금지 시험 |
| 자동 파이프라인 | throughput64 프로필, source fixture → 영어 기준 → 혼합 scalable 학습 → INT8 export → 온도 보정 → 평가 → 번들 완료 |
| 자동 재개 | 실행 도구 제한으로 최종 평가 단계에서 중단된 뒤 동일 명령으로 완료. 다음 재실행에서 15개 완료 단계 resumed_skip |
| 네이티브 대응 | 최종 smoke 번들 3표본의 최대 절대 logit 차이 3.725290298461914e-08 |
| 패키징 | wheel 빌드·별도 디렉터리 설치. 프로젝트 외부에서 torch/numpy/transformers/datasets/pyarrow/onnxruntime import를 차단한 C++ 4-block 추론 성공 |
| 정적 검사 | compileall, bash -n, git diff --check 통과 |

최종 smoke는 `mixed_candidate_passed_validation_gate`였다. 이는 색상 fixture의 검증 조건 통과일 뿐 영어·한국어 개선 증거가 아니다. 모든 smoke 결과는 시험용으로 표시되며 실제 모델을 배포하지 않았다.

단일 프로세스와 2-rank 전체 신경망 학습의 비교에서는 일부 파라미터에 약 2.35e-6 차이가 관측됐다. FP32 연산 순서의 영향까지 bitwise 동일하다고 주장하지 않는다. 별도 double-precision 전역 weighted-gradient 수학 시험과 동일 설정 내 CPU resume 시험은 각각 통과했다.

## 환경

Linux x86-64, CPython 3.13.5, PyTorch 2.10.0+cpu, NumPy, safetensors, pytest, C++17. CUDA 장치와 실제 Rust tokenizer/대형 사전학습 백본을 사용한 검증은 수행하지 않았다. GitHub connector로 원천 인터페이스를 확인했지만 컨테이너의 외부 다운로드가 제한되어 실제 공개 코퍼스 수집을 검증하지 못했다.

## 미실행·미완료를 구분

**미실행:** 실제 5M/11M 다운로드·본학습, CUDA/NCCL/BF16·fused AdamW의 속도/VRAM, 실제 4K/8K 이해력, 실기기 센서/표 데이터 held-out 성능, Julia/Laya/JevBench 실제 가중치 점수. JevBench 전체 upstream runner의 실데이터 실행도 아직 하지 않았으며 어댑터 계약 시험만 수행했다.

**미완료 구현:** 새 센서/표/token-interaction 그래프의 C++ export·커널, FSDP, GPU 동적 요청 배치 서비스, 전체 Auto5M Store의 병렬 ingest. 연구 trainer는 단일 장치 FP32·row-wise accumulation이며 text scalable trainer의 DDP와 다르다. 연구 모듈의 exact resume과 장문 전용 보정 학습 CLI는 후속 범위다.

지원 길이·modality를 수용하는 코드가 있다는 것과 그 범위의 품질이 승인됐다는 것은 다르다. `release_approved=false`를 유지하며, 낮은 확률을 이유로 정상 판단을 자동 null로 바꾸지 않는다.
