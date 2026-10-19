# NanoDecision2 확률 학습 설계

이 문서는 학습 목표·보정·판단 보류의 수학적 의미를 설명합니다. Jev의 비공개 RLCD를 재현한다는 주장이 아닙니다.

## 1. 조건부 확률 판단

상태 (s), 질문 (q), 선택지 집합 (C)가 주어질 때 모델은 (P(Y=k\mid s,q,C))를 예측합니다. 입력 선택지가 바뀌면 이 분포도 바뀔 수 있습니다. 정답 후보가 없을 수 있는 실제 업무에는 독립적으로 검증된 '해당 없음' 사례가 필요합니다.

- **Choice:** 실행 시 전달된 후보 사이의 확률분포
- **Noul:** false/true 확률, true의 확률을 별도로 반환
- **Score:** 순서 있는 레벨별 확률과 기대값

## 2. 로그 손실과 Brier

정답 분포 (q), 예측분포 (p)에 대해 학습 코드는 다음을 사용합니다.

[
L_{\mathrm{base}}=-\sum_k q_k\log p_k+\lambda_B\sum_k(p_k-q_k)^2
]

기본 (lambda_B=0.2)는 실험 시작값이지 측정된 최적값이 아닙니다. 실제 참분포 (pi)에서 관측된 일회 정답에 대해, 예측 (p)의 기대 초과 위험은

[
R(p)-R(\pi)=D_{\mathrm{KL}}(\pi\Vert p)+\lambda_B\lVert p-\pi\rVert_2^2\ge0.
]

따라서 이론적으로 진짜 분포가 최적입니다. 하지만 데이터 부족, 잘못된 라벨, 모델 표현의 제한, 분포 이동에서는 보정된 확률을 보장하지 않습니다.

## 3. 순서형 Score

순서가 있는 값 (v_1<\cdots<v_K)에서 누적분포 (F_p(j)=\sum_{k\le j}p_k)를 계산하고,

[
L_{\mathrm{ord}}=\sum_{j=1}^{K-1}\frac{v_{j+1}-v_j}{v_K-v_1}(F_p(j)-F_q(j))^2
]

를 추가합니다. 기본 계수는 0.2입니다. 가까운 점수와 먼 점수의 차이를 반영하되, 임의 순서인 Choice에는 적용하지 않습니다. 이는 유한 레벨의 정규화된 CRPS/RPS 계열입니다.

## 4. 언어 혼합과 영어 성능 보호

영어·한국어·기타 언어의 데이터를 함께 사용합니다. 한국어 향상이 영어 하락을 상쇄하도록 언어 평균 하나로만 체크포인트를 선택하지 않습니다. 영어 전용 기준 모델과 혼합학습 모델을 고정된 검증 질문에서 비교해 영어 정확도 및 NLL 회귀를 검사하고, 한국어도 같은 기준으로 확인합니다.

**이는 특정 검증셋에서의 관측 조건이며, 모든 실제 영어 입력에서 비열화를 증명하는 조건이 아닙니다.**

의미적으로 검토된 병렬문장 쌍에서만 예측분포의 Jensen–Shannon 발산을 학습에 선택적으로 추가할 수 있습니다. 단지 같은 정답 라벨을 가진 문장을 억지로 묶으면 진짜 분포를 왜곡할 수 있으므로 자동 활성화하지 않습니다.

## 5. 배포 후 Temperature scaling

INT8로 내보낸 **실제 배포 가중치**의 logits (z_i)에 대해 별도의 temperature 분할에서

[
T^*=\arg\min_{T>0}\sum_i\mathrm{CE}(q_i,\mathrm{softmax}(z_i/T))
]

를 맞춥니다. 양의 (T)는 argmax를 바꾸지 않으므로 정확도 개선 기법 자체가 아닙니다. 구현은 언어·문제 종류·후보 수 버킷별 보정을 지원하며 표본이 부족하면 전역 온도로 되돌아갑니다.

## 6. 원본 그룹 단위 Conformal prediction

같은 원본에서 파생된 여러 질문을 독립 사례 여러 개로 세지 않습니다. 그룹 (g)의 실제 정답 (y_i)에 대한 nonconformity는

[
S_g=\max_{i\in g}(1-p_i(y_i))
]

입니다. (G)개 calibration 그룹에서 (k=\lceil(G+1)(1-\alpha)\rceil)번째 순서통계량을 임곗값으로 정하고, 그 임곗값 아래의 후보를 예측 집합으로 반환합니다. (k>G)이면 모든 후보를 포함하는 집합으로 처리합니다.

**교환 가능성 등의 조건에서 그룹 전체 정답 포함률에 관한 주변적 결과**입니다. 개별 정답이 90% 안전하다는 뜻이나, 보류 없이 수락한 답의 정확도가 90%라는 뜻은 아닙니다. 낯선 분야·언어·악성 입력에 대한 보장은 추가되지 않습니다.

## 7. 행동 비용

확률이 주어졌을 때 기대 비용

[
a^*=\arg\min_a\sum_y p_yC(a,y)
]

로 행동을 선택할 수 있습니다. 이 인터페이스는 계산 결과만 반환하며 외부 도구를 직접 실행하지 않습니다. 확률 품질, 비용 행렬과 별도의 권한 검사가 필요합니다.

## 8. 데이터·평가 격리

학습, 체크포인트 선택, temperature, conformal, 최종 테스트를 분리합니다. 원본 그룹과 번역 관계가 평가에 누출되지 않도록 검사합니다. MS MARCO 등의 약한 라벨을 인간이 확정한 참이라고 간주하지 않습니다. Julia·Laya 비교에서는 같은 상태·질문·후보·정답을 사용하고 실패율도 함께 보고합니다.

**학습이 완료됐다는 것과 Julia/Laya를 능가한다는 것은 별개입니다.** 실제 장문·한국어/영어·다중 질문 처리량은 별도의 보류 평가가 필요합니다.

## 참고

- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation.*
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- Angelopoulos & Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
- Gururangan et al. (2020), [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964)

기본 구현의 정확한 옵션과 검증 범위는 `nanodecision2/losses.py`, `nanodecision2/probability.py`, `nanodecision2/training.py`, `reports_v031/VERIFIED.md`를 확인하세요.
