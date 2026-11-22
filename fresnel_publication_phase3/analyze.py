"""Audit saved proofs/target flags and produce the matched holdout report."""
import csv,itertools,json,statistics
from collections import Counter
from fractions import Fraction as F
from math import sqrt
from experiment import D,pilot,stats

def save(name,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (D/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,cols)
        w.writeheader()
        w.writerows(rows)

def md(name,lines):
    (D/name).write_text('\n\n'.join(lines)+'\n',encoding='utf8')

def main():
    manifest=json.loads((D/'holdout_manifest.json').read_text())
    results,diag,bench=[],[],[]
    increases=[]
    for i,c in enumerate(manifest['cases'],1):
        for model in manifest['models']:
            common=None
            methods={}
            for method in 'ABC':
                dest=D/'runs'/f'{i:02}_{model}_{method}'
                s=json.loads((dest/'search.json').read_text())
                tag=dict(case=i,**dict(zip(manifest['order'],c)),model=model,method=method)
                counts=0
                seen={}
                for r in s['attempts']:
                    n,b=r['degree'],r['bits']
                    line=dict(**tag,n=n,bits=b,worker_status=r['status'],reason=r['reason'],seconds=r['seconds'],peak_mib=r.get('peak_worker_mib'))
                    diag.append(line)
                    if not r.get('certificate'): continue
                    path=dest/r['certificate']
                    payload=json.loads(path.read_text())
                    if payload['verification'].get('verified') is not True: raise ValueError(path)
                    rec=payload['certificate']
                    l,u,w,rel,rok,aok=pilot.metric(rec,F('0.01'),F('1e-5'))
                    common=(l,u) if common is None else (max(common[0],l),min(common[1],u))
                    if common[0]>common[1] or (rok,aok)!=(r['relative_met'],r['absolute_met']): raise ValueError(path)
                    g,d,M,eps=stats(rec)
                    gl,gu=map(pilot.rational,rec['obligations']['gram']['evidence']['G'])
                    line.update(width=float(w),M=float(M),kernel_epsilon=float(eps),field_delta=float(d),gram_squared_width=float(g),gram_lower=float(gl),gram_upper=float(gu),gram_norm_width=sqrt(float(gu))-sqrt(float(max(gl,0))),lower_clipped=l==0,certificate=str(path.relative_to(D)))
                    for axis in ('degree','bits'):
                        keys=[(nn,bb) for nn,bb in seen if (bb==b and nn<n if axis=='degree' else nn==n and bb<b)]
                        if not keys: continue
                        key=max(keys)
                        old=seen[key]
                        line['previous_'+axis]=key[0 if axis=='degree' else 1]
                        for field in ('M','field_delta','gram_squared_width','width'):
                            line[axis+'_change_'+field]=float(dict(M=M,field_delta=d,gram_squared_width=g,width=w)[field]-old[field])
                        if axis=='degree' and w>old['width']:
                            increases.append(line)
                    seen[n,b]=dict(M=M,field_delta=d,gram_squared_width=g,width=w)
                    counts+=1
                best=s['best'] or {}
                out=dict(**tag,status=s['status'],reason=s['reason'],verified_candidates=counts,candidates=len(s['attempts']),seconds=s['seconds'],peak_worker_mib=s['peak_worker_mib'])
                for key in ('degree','bits','lower','upper','absolute_width','relative_width','absolute_met','relative_met'):
                    out[key]=best.get(key)
                results.append(out)
                methods[method]=out
            row={k:v for k,v in tag.items() if k!='method'}
            for method,r in methods.items():
                for key in ('status','seconds','candidates','peak_worker_mib','absolute_width','relative_width'):
                    row[f'{method}_{key}']=r[key]
            row['AB_both_met']=all(methods[k]['status']=='target_met' for k in 'AB')
            row['A_over_B_time_if_both_met']=methods['A']['seconds']/methods['B']['seconds'] if row['AB_both_met'] else None
            bench.append(row)
    save('holdout_results.csv',results)
    save('interval_width_diagnostics.csv',diag)
    save('matched_benchmark.csv',bench)
    totals={}
    for m in 'ABC':
        rr=[r for r in results if r['method']==m]
        totals[m]=dict(met=sum(r['status']=='target_met' for r in rr),verified=sum(r['verified_candidates']>0 for r in rr),seconds=sum(r['seconds'] for r in rr),peak=max(r['peak_worker_mib'] or 0 for r in rr),attempts=sum(r['candidates'] for r in rr))
    paired=[r for r in bench if r['AB_both_met']]
    br=[r for r in results if r['method']=='B']
    fail=[r for r in br if r['status']!='target_met']
    lines=['# 목표 미달 및 폭 진단',f'B의 목표 미달은 {len(fail)}/48건이다. 아래 수치는 각 층에서 미달한 건수 / 시도 건수다.']
    for key in ('nf','ar','p','theta','model'):
        lines.append(key+' — '+', '.join(f"{v} ({sum(r[key]==v for r in fail)}/{sum(r[key]==v for r in br)})" for v in sorted(set(r[key] for r in br))))
    lines+=['B 미달 사례 (조건 번호, 모델, 종료 이유, 최선 절대폭, 상대폭)']
    lines+=['- '+', '.join(str(r[k]) for k in ('case','model','reason','absolute_width','relative_width')) for r in fail]
    lines.append(f'같은 방법·조건·비트에서 차수를 올리자 최종 폭이 증가한 비교는 {len(increases)}건이다. CSV의 차이 열은 동일 비트 또는 동일 차수의 직전 관측값을 뺀 값이다. 빈 칸은 해당 짝이 없음을 뜻한다.')
    if increases:
        for r in increases:
            lines.append(f"- {r['case']} {r['model']} {r['method']}, n={r['previous_degree']}→{r['n']}, b={r['bits']}. Δ폭 {r['degree_change_width']:.3g}, ΔM {r['degree_change_M']:.3g}, Δδ {r['degree_change_field_delta']:.3g}, ΔGram 폭 {r['degree_change_gram_squared_width']:.3g}.")
    lines.append('최종 구간은 Gram 제곱근의 양끝에 장 나머지 δ를 더하거나 빼고 하한을 0으로 제한한다. CSV의 kernel_epsilon은 커널 보간 오차, field_delta는 물리 계수가 반영된 장 오차다. Gram 제곱 구간 폭의 단위는 장 폭과 다르다. 인증서는 정확한 이진 끝점을 저장하며 CSV 진단 차이는 표시용 근사치다.')
    groups=[r for r in diag if 'bits_change_gram_squared_width' in r]
    lines.append(f'같은 차수에서 정밀도만 올린 비교 {len(groups)}건 중 Gram 폭 감소 {sum(r["bits_change_gram_squared_width"]<0 for r in groups)}건. 특정 산술 연산이 병목이라는 인과 추론은 이 집계로 검증되지 않는다.')
    md('failure_analysis.md',lines)
    lines=['# Phase 3 독립 홀드아웃 평가','24개 물리 조건 × 두 잔차. Phase 2 파일과 선택기 해시는 holdout_manifest.json에 있다. A는 원본 탐색, B는 사전 고정 선택, C는 64차·512비트 고정이다. 목표는 상대폭 1%, 절대폭 1e-5를 동시에 충족하는 것이다.']
    for m,t in totals.items():
        lines.append(f"{m} — 목표 달성 {t['met']}/48 ({100*t['met']/48:.1f}%), 검증 구간 보유 {t['verified']}/48, 후보 {t['attempts']}개, 총 {t['seconds']:.2f}초, 최고 관측 작업자 메모리 {t['peak']:.1f} MiB.")
    if paired:
        a,b=(sum(r[f'{m}_seconds'] for r in paired) for m in 'AB')
        lines.append(f'양쪽 모두 목표를 달성한 {len(paired)}건의 누적 wall-clock은 A {a:.2f}초, B {b:.2f}초다. 개별 A/B 시간비 중앙값은 {statistics.median(r["A_over_B_time_if_both_met"] for r in paired):.2f}다. 방법 순서를 순환한 단일 직렬 실행이며 CPU·컴퓨터 간 속도 우위로 일반화할 수 없다.')
    lines+=['독립 검증 통과는 고정된 Theorem 1과 Arb를 신뢰 기반으로 둔 해당 입력 구간의 수치 인증이다. 두 폭 목표의 통과 여부는 이진 유리수로 별도 판정했다. 사례별로 세 방법의 모든 승인 구간이 공통점을 갖는지도 확인했다.',
    '논문에서 제시할 수 있는 기여는 동일한 연속 원판 복소장 RMS에 대해 초타원 모멘트로 두 물리 모델 잔차의 양측 구간을 계산하고, 사전 고정한 선택기를 미사용 조건에서 검증한 실험이다. 최초성이나 임의 개구의 보장은 이번 실험으로 입증되지 않는다.',
    '투고 전에는 새 극단 조건에서 더 큰 예산(특히 96·128차)을 사전 고정한 두 번째 홀드아웃, 동일 목표 성공 사례의 반복·교차 하드웨어 시간 측정, 독립 구현 또는 고정밀 직접 적분 대조가 우선이다. 성공률은 정해진 격자의 기술통계이며 무작위 모집단 추정치가 아니다.',
    '재현 시 Phase 2의 Python 3.14.7 및 python-flint 0.9.0 환경에서 별도 사본을 사용한다. 원래 디렉터리 구조와 holdout_manifest.json을 보존하고 빈 runs 경로에서 python -B experiment.py, 이어서 python -B analyze.py를 실행한다. Windows Job Object의 프로세스별 commit 상한은 512 MiB, 보고 메모리는 작업자 peak working set이다. 시간은 검색 전체이며 종료 결과 JSON 기록과 후처리는 제외한다.']
    md('phase3_publication_assessment.md',lines)
    from diagnose import append_diagnosis
    append_diagnosis(results)
    print(totals,len(paired),len(increases))

if __name__=='__main__': main()

