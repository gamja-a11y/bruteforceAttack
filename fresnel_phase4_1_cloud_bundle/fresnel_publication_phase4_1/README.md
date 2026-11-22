# Phase 4.1 Ubuntu 실행

이 폴더를 기존 Phase 2–4 및 수정 인증기 폴더와 같은 연구 루트에 둔다. 원본 파일과 경로가 준비된 Ubuntu/WSL mp 환경에서 실행한다.

~~~bash
cd "/mnt/c/Users/smhsu/OneDrive/문서/ChatGPT/프레즈넬 공부 및 뭐시기저시기/fresnel_publication_phase4_1"
conda activate mp
python -B prepare.py
python -B run.py --smoke
python -B run.py
python -B summarize.py
~~~

prepare는 기존 명세·개발 CSV의 물리 좌표를 유리수로 대조한다. 초기 24조건 중 중복은 같은 NF·AR·p의 θ=0.21, 0.31, 0.22, 0.32 순서에서 처음 비어 있는 값으로 대체한다. 기존 manifest와 해시가 있으면 검증만 수행한다. smoke는 기존 개발 case 9 Padé 한 조건에서 B/C의 128차·1024비트 인증을 시행하고 각 기존 구간과 교집합을 검사한다. 본 실행도 시작 전에 smoke의 인증서를 검증한다.

본 실험은 24개 물리 조건 × 두 모델 × 두 방법의 96회 직렬 실행이다. B는 ρ=2, C는 기존 512비트 contour 스캔과 동결 choose 함수를 호출하며 이번에 허용되지 않은 64·80·96차를 done으로 전달한다. 최종 발급·검증은 양쪽 모두 128차·1024비트다. 매 작업은 시작·탐색·발급·검증·자식 종료를 포함해 90초, Linux RLIMIT_AS 768 MiB로 제한한다. peak는 wait4의 자식 최대 RSS다. 원인이 불확실한 네이티브 프로세스 종료는 worker_exit와 stderr로 보존한다.

run.py를 다시 실행하면 완료 기록의 해시를 검사하고 인증서는 동결 입력과 재선택한 ρ로 재검증한다. 기록된 실패도 보존한다. 중단되어 영수증이 없는 폴더는 interrupted 이름으로 보존하고 계산을 이어간다. 영수증은 로컬 변경 감지용 SHA-256이다. 재검증 시간은 최초 측정에 합산하지 않는다.

summarize는 인증서를 다시 검증해 results.csv, paired.csv, summary.json을 만든다. 감소율은 100×(B−C)/B이며 분모가 0이면 비워 둔다. 미실행·불완전 사례는 성공/실패 전환에서 제외한다. 시간 초과 등으로 인증서가 없을 때에는 알 수 없는 필드를 비워 둔다. smoke는 본 실험 통계에 들어가지 않는다.
