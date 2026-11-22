# 기여 주장과 심사 리스크

**논문에 쓸 수 있는 문장.** 면적 보존 초타원과 중심 원판에서 RS-I–표준 Fresnel 및 RS-I–Padé의 절대 복소장 RMS 잔차를 동일한 정규화로 양측 포괄한다. 개구 모멘트로 적분한 장 다항식과 원판 Gram 노름을 결합해 복소 상쇄를 보존한다. 독립적으로 재계산한 구간에서만 사용자가 지정한 폭의 충족 여부를 결정한다. 이는 구현하고 시험한 기여의 범위다. 세계 최초라는 주장은 이번 검색으로 입증되지 않았다.

|주장|이미 확인한 선례|구별할 부분|
|---|---|---|
|요청 정확도로 회절 전파 계산|Lewis 등의 정리 3.3에 장 계산 오차 제어가 있다. sinc 전파에도 목표 오차에 따른 파라미터 선택이 있다.|RS-I 해의 수치 계산 오차와 두 물리 모델 사이의 연속 잔차 노름은 다르다. [Lewis 등](https://pmc.ncbi.nlm.nih.gov/articles/PMC3780824/) · [Cubillos & Jimenez](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-39-8-1403)|
|개구 기하와 Fresnel 유효성 연결|Mezouari & Harvey가 위상·축상 진폭 오차 기반 유효 구역을 연구했다.|본 연구는 원판 평균 복소장 RMS의 두 끝점을 제공한다. [저자 원문](https://home.eps.hw.ac.uk/~ceearh2/Publications/2003/JOPA%20Fresnel-Fourier%20diffraction.pdf)|
|다항식 회절과 개구 모멘트|Chebyshev 회절 적분과 다각형 Fourier 모멘트의 해석적 계산에 각각 선례가 있다.|Lobatto 보간, 감마·베타 공식, Gram 항등식, 삼각부등식은 기존 수학이다. [Kovalev 등](https://www.ripublication.com/ijaer17/ijaerv12n23_39.pdf) · [Wuttke](https://arxiv.org/pdf/1703.00255)|
|적응형 엄밀 구적과 위상 보존|검증 구적은 이미 복소 타원과 차수 조절을 이용한다. 복소 적분과 NUFFT도 위상 간섭을 계산한다.|상쇄를 보존하는 개구 적분과 연속 원판 잔차의 양측 구간, 독립 재계산을 한 실험에서 연결한 것이 이 연구의 구체적인 차별점이다. [Johansson](https://arxiv.org/pdf/1802.07942) · [Barnett](https://arxiv.org/pdf/2010.05978)|

## 남은 리스크

발급기와 검증기는 동일한 FLINT/Arb와 문서의 수학 정리를 신뢰한다. 허용 오차를 만족하는 최소 계산량, 모든 형상에서의 종료, 보편적 가속 및 문헌상의 우선권은 미입증이다. 샘플에서 얻은 두 형상 사이의 할선 기울기는 점별 미분의 인증이 아니다. 범위 밖의 비균일 조명, 비중심 원판, 임의 개구 및 Maxwell 문제에 적용하려면 추가 증명이 필요하다. 투고 전에 JOSA A 2022의 오차 정리와 입력 정칙성 가정을 원문으로 대조해야 한다.

