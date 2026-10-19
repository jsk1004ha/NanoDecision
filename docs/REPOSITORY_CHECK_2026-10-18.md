# 저장소 정리·재검증 — 2026-10-18

대상: 제공된 `NanoDecision2_0.3.1_Auto5M.zip`의 소스.

## 실제 실행

- CPython 3.13, Linux x86-64, CPU 환경에서 C++ 확장 빌드 완료.
- 로컬 소스를 `pip install --no-deps --no-build-isolation -e .`로 설치.
- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m pytest -q`: **90 passed in 6.97s**.
- 이 검증 환경에는 torch/numpy/safetensors/pytest가 이미 설치되어 있었으며,
  tokenizers/transformers는 설치되어 있지 않아 기존 byte-token fixture 경로를 사용했습니다.
- 첫 미설치 상태의 시험은 하위 프로세스가 패키지를 찾지 못해 1개 실패했습니다.
  문서의 설치 전제에 맞춰 editable 설치 후 90개가 모두 통과했습니다.
- `pytest_repository_20261018.log`에 최종 시험 출력을 보존했습니다.

## 이번 변경 범위

README.md 작성, Git 제외 규칙·텍스트 줄바꿈 규칙 추가, 저장소 URL과 README
패키징 메타데이터 정리, 기존 Apache LICENSE 보존 및 원 배포본 MIT 고지 분리 보존.
모델 구조·데이터 레시피·학습/추론 Python·C++ 코드는 변경하지 않았습니다.

## 검증 범위의 제한

이번 시험은 실제 대규모 데이터 다운로드·학습, CUDA 처리량, 한국어/영어 성능,
Julia/Laya 비교를 수행한 결과가 아닙니다. 저장소 게시의 성공 여부를 나타내지도 않습니다.
GitHub 서버에 대한 실제 변경은 별도로 확인해야 합니다.
