# 논문 기여와 남은 검증

|기여|기존에 알려진 내용|이번 연구에서 검증한 내용과 남은 범위|
|---|---|---|
|연속 원판 RMS 양측 인증|RS 전파와 Fresnel 근사의 유효 범위는 오랜 연구 주제다. Lewis 등은 주어진 허용오차로 복소 전파장을 계산한다.|동일 초타원·ROI에서 두 잔차의 절대 복소장 RMS를 각각 양측으로 포함한다. 원판 평균, 단위 입사장, 스칼라 모델이 주장의 범위다.|
|상쇄를 보존하는 모멘트|Bernstein·Chebyshev 근사, Dirichlet 모멘트, 다항식 Gram 항등식은 기존 수학이다. Barnett은 일반 개구의 복소 Fresnel 장을 면적 구적과 NUFFT로 계산한다.|초타원 모멘트를 먼저 복소수로 합한 뒤 원판 위 제곱을 적분하고 교차항을 보존했다. 임의 개구 또는 많은 목표점에서는 기존 전파법이 더 일반적이다.|
|목표 정확도 선택|엄밀한 구적에서 사용자 허용오차, 분할 및 계산 제한은 기존 도구다. Johansson은 ball arithmetic을 사용한 인증 구적을 다룬다.|동일한 상대·절대 목표를 정확한 이진 끝점으로 검사하고, 실패를 포함한 독립 홀드아웃을 공개했다. 모든 입력에서 성공한다는 보장은 없다.|
|ρ와 차수의 결합 선택|Wang과 Huybrechs는 Chebyshev 계수의 상대 정확도를 위해 타원 크기를 선택했다. Bernstein 타원에서의 계수 감쇠도 고전적이다.|여기서 선택 대상은 실제 계수 정확도가 아니라 물리적 잔차장의 보간 나머지 예산이다. 호별 Arb 상한을 사용한 유한 후보 비교와 인증 차수 변화가 기여다. 연속 최적성 및 최초성은 미확인이다.|
|인증서와 재검증|Arb를 이용한 엄밀한 수치 계산과 독립 검산 자체는 기존 방법이다.|동일한 명세에서 Newton/DCT-I와 서로 다른 축 수축으로 수학적 증거를 재계산하고, 변조와 플랫폼 재현성을 시험했다. 발급기와 검증기는 FLINT/Arb와 해석적 정리를 공유한다.|

출판 시 가장 중요한 추가 실험은 고NF 경계에서 **ρ 검색으로만 추가 성공한 사례의 재현성**을 다른 NF·축비와 보지 않은 조건에서 확인하는 것이다. 계산 비용에 대해서는 여러 대표 조건과 다른 CPU에서 직렬 기준의 반복 측정이 필요하다. 한 시스템의 시간이나 개발용 후보 중 사후 최솟값으로 보편적 속도·최초성·저널 등급을 주장하기에는 근거가 부족하다.

문헌 확인 범위는 Lewis 등의 본문 및 기존 Phase 2 문헌표, Trefethen의 해석함수 계수·보간 정리, Johansson, Barnett, Wang–Huybrechs의 공개 서지·초록을 포함한다. 후자의 초록 확인으로 모든 정리의 적용 범위를 판정하지 않았다. 투고 전에는 validated spectral methods, optics model discrepancy norm, ellipse selection의 체계적 검색이 필요하다. [Lewis 등](https://pmc.ncbi.nlm.nih.gov/articles/PMC3780824/) · [Trefethen, Chapter 8](https://people.math.ethz.ch/~hiptmair/Seminars/RAP_22/TRE13.pdf) · [Johansson](https://arxiv.org/abs/1802.07942) · [Barnett](https://arxiv.org/abs/2010.05978) · [Wang & Huybrechs](https://doi.org/10.1093/imanum/drw039)

