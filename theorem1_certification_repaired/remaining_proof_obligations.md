# 재감사의 범위와 남은 증명 의무

## 이번에 확인한 내용

발급기의 공개 진입점은 `repaired_certification.certify` 및 이를 호출하는 명령행이다. 두 모델이 이 경로를 공유한다. 단위 입사, 면적 100π의 축 정렬 초타원, 양의 실수 파장·거리와 중심 원판 면적 평균을 전제로 한다. p<2와 AR<1을 지원한다. 영역 포함 비교가 불분명하거나 Bernstein 타원의 해석성 또는 유한 상한을 입증할 수 없는 입력은 거부한다. 넓은 인증 구간의 정밀도와 발급 여부는 별개다.

* 15개 정상 인증서를 생성하고 다른 구현으로 재검산했다. 96차·768비트의 기준 Padé는 기존 v6 구간과 독립 감사 및 표준 경로의 Padé 구간 모두와 겹친다. 기준 Fresnel은 기존 동일 모델 구간과 겹친다.
* 과거 테스트 본문 `kernel_tests`, `polynomial_tests`, `adversarial`, `comparison_tests`를 보관 사본에서 읽어 새 호출 인터페이스에 연결했다. 예외를 행으로 기대하는 옛 테스트에 한해 어댑터가 예외를 실패 행으로 옮긴다. 새 공개 API의 실제 예외는 별도 테스트에서 직접 확인했다. 과거 재현기의 실행 프레임·지역변수 이름 추적은 이전 세 구현 전용이다. 새 구현에서는 명시적 의무 기록과 독립 검증이 그 역할을 맡는다.
* 독립 검증기는 해시, 입력, 10개 증명 단계의 수치 증거를 확인한다. 각 호의 상한을 재계산하고 다른 차수 기저 변환, 소스 합산 순서, Gram 곱을 사용하여 계산한 더 높은 정밀도의 구간 전체가 기록된 G 안에 들어가는지 검사한다. 끝점만 맞는 허위 Gram이나 누락된 호를 탐지하는 시험을 포함한다.
* 64비트를 최소 작업 정밀도로 선택했다. 80비트에서도 안전하게 넓은 구간을 반환하는 사례를 포함했다. NF, θ, AR 등 수치는 유한 양수의 정수·실수·Decimal·십진 문자열 입력을 받는다. float 입력은 그 이진 부동소수점의 실제 값을 뜻하므로 논문 조건은 문자열로 입력하는 편이 재현하기 쉽다.

## 신뢰 기반 및 미검증 항목

1. 등호를 포함한 수학 부등식은 문서의 증명과 그 가정에 의존한다. CPython, python-flint 0.9.0, FLINT 3.6.0의 복소·실수 볼 연산과 초월함수의 포함성은 신뢰 기반이다. 이 라이브러리 자체와 수학 정리를 형식 증명기로 재검증하지 않았다. [FLINT Arb documentation](https://flintlib.org/doc/arb/index.html)
2. 독립 검증기와 발급기는 같은 Arb 백엔드를 사용한다. 타원 호의 상한에는 같은 수학식을 사용하므로 두 프로그램에 공통인 수식 오류는 문서의 해석성 증명과 기호 항등식 감사가 담당한다. 저장된 소스 해시는 로컬 검토본과의 동일성 확인용이다. 발행자의 전자서명 또는 원격 파일의 출처 인증은 별도 의무다.
3. 검증기 CLI는 인증서가 기술하는 입력을 재생한다. 특정 실험의 의뢰와 연결할 때는 `verify(record, expected=...)`에 기대하는 전체 입력을 독립적으로 제공해야 한다. 이번 기준 재감사는 이를 수행했다.
4. 성공·실패 사례는 명시된 판정 범위를 시험한다. 임의 정밀도·입력 크기의 자원 소요, 다른 백엔드로의 이식, 다른 조명·비중심 ROI·전자기 경계조건 및 속도 우위는 검증 범위 밖이다.
5. 개발 중 고정 면적 검사를 내부 모멘트 함수에 적용했을 때 과거의 단위 원판 오라클이 중단됐다. 최종 구현은 일반 모멘트 함수에서 주어진 축의 면적을, 발급 진입점에서 100π를 확인한다. Q=7.99 근처에서는 구간 `1/(t**3)`이 불필요하게 넓어져 동일한 값인 `(1/t)**3`으로 계산했다. 수정 후 이전 단위 원판 검사와 Q=7.99, 8, 8.01의 경계 검사를 모두 다시 통과했다.

## 재현

현재 Windows 환경에서는 이 폴더에서 다음을 실행한다. 계산 결과는 새로운 폴더에 쓴다.

~~~powershell
.\run_repaired.ps1 -OutputDirectory replay_01
~~~

같은 의존성이 준비된 Python에서는 아래 명령으로 생성기와 검증기를 별도 프로세스에서 사용할 수 있다. 전체 과거 감사 재실행에는 NumPy, SciPy, SymPy, mpmath도 필요하다. 시험 환경의 버전은 previous_audit/final_certificate.json에 있다.

~~~text
python -B run_reaudit.py --out replay_01
python -B repaired_certification.py --model Fresnel --degree 96 --bits 768 --out fresnel_new.json
python -B -O repaired_independent_verification.py fresnel_new.json --out fresnel_checked.json
python -B -O repaired_independent_verification.py corrected_certificate.json --out new_verifications.json
~~~

