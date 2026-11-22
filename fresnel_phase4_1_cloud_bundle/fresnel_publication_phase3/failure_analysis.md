# 목표 미달 및 폭 진단

B의 목표 미달은 18/48건이다. 아래 수치는 각 층에서 미달한 건수 / 시도 건수다.

nf — 1 (0/16), 10 (16/16), 3 (2/16)

ar — 1 (8/24), 4 (10/24)

p — 10 (9/24), 4 (9/24)

theta — 0.1 (10/24), 0.3 (8/24)

model — Fresnel (8/24), Pade (10/24)

B 미달 사례 (조건 번호, 모델, 종료 이유, 최선 절대폭, 상대폭)

- 7, Pade, candidate limits exhausted, 6.645163212972771295369E-10, 0.03366451168421382490938

- 9, Pade, candidate limits exhausted, 0.0001797025176303860638106, 2

- 9, Fresnel, candidate limits exhausted, 0.0001168151494638915764734, 0.008044456659532754213829

- 10, Pade, candidate limits exhausted, 0.00002710606585293736573997, 2

- 10, Fresnel, candidate limits exhausted, 0.0001680374706873756660045, 0.001621135676737602995965

- 11, Pade, candidate limits exhausted, 233374674378.1028442934, 2

- 11, Fresnel, candidate limits exhausted, 191222256512.6996690770, 2

- 12, Pade, candidate limits exhausted, 718966386.3118301996510, 2

- 12, Fresnel, candidate limits exhausted, 118542982815.9714893045, 2

- 19, Pade, candidate limits exhausted, 3.215601545286144652712E-10, 0.01440732115154417074463

- 21, Pade, candidate limits exhausted, 0.001887649344997034728453, 2

- 21, Fresnel, candidate limits exhausted, 0.001339762964703643180106, 0.1274149049638488883060

- 22, Pade, candidate limits exhausted, 0.0004388070976759995667566, 2

- 22, Fresnel, candidate limits exhausted, 0.004375129073541905508670, 0.05743777455056744279476

- 23, Pade, candidate limits exhausted, 23879187106.19702793446, 2

- 23, Fresnel, candidate limits exhausted, 18425116745.25072168248, 2

- 24, Pade, candidate limits exhausted, 140210732.1372626351461, 2

- 24, Fresnel, candidate limits exhausted, 17303177685.98427013339, 2

같은 방법·조건·비트에서 차수를 올리자 최종 폭이 증가한 비교는 0건이다. CSV의 차이 열은 동일 비트 또는 동일 차수의 직전 관측값을 뺀 값이다. 빈 칸은 해당 짝이 없음을 뜻한다.

최종 구간은 Gram 제곱근의 양끝에 장 나머지 δ를 더하거나 빼고 하한을 0으로 제한한다. CSV의 kernel_epsilon은 커널 보간 오차, field_delta는 물리 계수가 반영된 장 오차다. Gram 제곱 구간 폭의 단위는 장 폭과 다르다. 인증서는 정확한 이진 끝점을 저장하며 CSV 진단 차이는 표시용 근사치다.

같은 차수에서 정밀도만 올린 비교 0건 중 Gram 폭 감소 0건. 특정 산술 연산이 병목이라는 인과 추론은 이 집계로 검증되지 않는다.

## 최선 후보의 나머지 예산

아래 절대폭 불가능 판정은 δ > 1e-5이다. 상대폭 불가능 판정은 2δ > 0.01U이다. 양의 하한에서는 폭 ≥ 2δ이고 상대폭 ≥ 폭/U이며, 하한 0은 규칙상 미달이다. 두 판정은 충분조건이다.

|조건|모델|δ만으로 절대폭 미달|현재 U에서 δ만으로 상대폭 미달|장 δ|Gram 제곱 폭|하한 0|
|---|---|---|---|---:|---:|---|
|7|Pade|False|True|3.323e-10|4.842e-106|False|
|9|Pade|True|True|0.0001797|3.848e-99|True|
|9|Fresnel|True|False|5.841e-05|2.979e-92|False|
|10|Pade|True|True|2.073e-05|9.015e-96|True|
|10|Fresnel|True|False|8.402e-05|1.671e-90|False|
|11|Pade|True|True|2.334e+11|2.012e-94|True|
|11|Fresnel|True|True|1.912e+11|1.877e-88|True|
|12|Pade|True|True|7.19e+08|7.935e-91|True|
|12|Fresnel|True|True|1.185e+11|6.469e-88|True|
|19|Pade|False|True|1.608e-10|5.508e-106|False|
|21|Pade|True|True|0.001888|5.393e-98|True|
|21|Fresnel|True|True|0.0006699|3.94e-91|False|
|22|Pade|True|True|0.0004339|3.501e-95|True|
|22|Fresnel|True|True|0.002188|5.264e-90|False|
|23|Pade|True|True|2.388e+10|3.735e-94|True|
|23|Fresnel|True|True|1.843e+10|5.563e-88|True|
|24|Pade|True|True|1.402e+08|9.145e-91|True|
|24|Fresnel|True|True|1.73e+10|1.465e-87|True|
