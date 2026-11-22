# 재현

동일한 디렉터리 구조에서, 원본을 보존한 별도 연구 폴더 사본을 사용한다. Python 3.14.7, python-flint 0.9.0, FLINT 3.6.0, matplotlib가 필요하다. Windows PowerShell에서 이 PC의 실행기는 다음과 같다.

~~~powershell
$py="$env:USERPROFILE/AppData/Local/Python/pythoncore-3.14-64/python.exe"
$env:PYTHONPATH="$env:USERPROFILE/AppData/Local/Temp/codex-fresnel-flint;$env:USERPROFILE/AppData/Local/Temp/codex-meep-validation-deps"
cd fresnel_publication_phase4
~~~

새 실험 재현에는 src, audits와 reports의 Python 코드, 빈 records·experiments·figures 폴더가 필요하다. 이미 등록된 명세와 결과가 있는 배포 사본에서 그대로 재실행하면 중복 파일 생성을 거부할 수 있다. build_copies.py는 두 확장 소스의 생성 내역이며 그대로 배포한 두 소스를 사용할 때는 실행할 필요가 없다.

~~~powershell
& $py -B src/common.py before
& $py -B src/compute.py
& $py -B src/integrity.py
& $py -B src/development.py
& $py -B src/dev_policies.py
& $py -B src/freeze.py
& $py -B src/experiment.py holdout
& $py -B src/experiment.py repeated
& $py -B audits/replay_contours.py
# Ubuntu의 동일 버전 FLINT 환경에서 Phase 4 폴더 기준으로 실행
wsl -d Ubuntu -- bash -lc '~/miniforge3/envs/mp/bin/python -B audits/wsl_probe.py'
wsl -d Ubuntu -- bash -lc '~/miniforge3/envs/mp/bin/python -B audits/wsl_regeneration.py'
& $py -B src/common.py after
& $py -B audits/aggregate.py
& $py -B reports/publication.py
& $py -B reports/diagnostics.py
~~~

이 실행에서는 개발 후보 그리드와 개발 정책 실험을 일부 동시에 수행했다. 실험 시간은 매번 바뀌므로 개발 결과 CSV를 포함하는 해시도 새 재현본에서는 달라진다. 최초 동결 명세로 평가하려면 제공된 해시와 소스를 보존하고 새 결과 디렉터리에서 실행한다. Windows 메모리 제한은 작업자 프로세스별 Job Object commit 한도이며 peak_mib는 peak working set이다. Linux 메모리 제한 경로는 RLIMIT_AS로 구현되어 있으며 이번 Ubuntu 검증은 수치 재생만 수행했다.

records 아래 개별 JSON에는 expected, certificate, verification이 있다. 재검증할 때는 independently constructed expected를 넘긴다. src/rho_certifier.py의 certify는 수학적 원시 기록을 반환하고, src/compute.py의 evaluate가 정확한 목표 플래그를 추가해 src/rho_independent_verifier.py의 check를 호출한다. 원시 수학 기록만을 정확도 성공 인증서로 사용하지 않는다. 실패·시간 초과는 experiments CSV와 records의 result.json에 보존한다. 동일 수치 대상의 개발 그리드는 차수별 지정 정밀도를 사용하고 96차에서만 512·768·1024비트를 모두 비교한다.

