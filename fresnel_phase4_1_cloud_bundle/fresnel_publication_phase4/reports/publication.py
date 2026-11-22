from pathlib import Path
import sys,statistics,itertools
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def save(name,lines): (D/'reports'/name).write_text('\n\n'.join(lines)+'\n',encoding='utf8')
def ratio(a,b):return float(F(a)/F(b))
def main():
    grid=loadcsv(E/'development_results.csv');sw=loadcsv(E/'rho_sweep_results.csv')
    hold=loadcsv(E/'holdout_results.csv');repeat=loadcsv(E/'repeated_runtime.csv')
    dp=loadcsv(E/'development_policy_comparison.csv')
    key=lambda r:(r['case'],r['model'])
    groups=sorted(set(map(key,hold)),key=lambda k:(int(k[0]),k[1]))
    matched=[]
    for k in groups:
        rr={r['method']:r for r in hold if key(r)==k}
        row=dict(case=k[0],model=k[1],nf=rr['A']['nf'])
        for m,r in rr.items():
            for fld in ('met','wall_s','n','bits','rho','width','relative','over_target_100x'):row[m+'_'+fld]=r[fld]
        for a,b in ('AB','AC','BC'):
            row[a+b+'_common']=rr[a]['met']==rr[b]['met']=='True'
        matched.append(row)
    writecsv(E/'matched_benchmark.csv',matched)
    def line(s):return sum(r['met']=='True' for r in s)
    summary=[]
    for m in 'ABC':
        rr=[r for r in hold if r['method']==m]
        good=[float(r['wall_s']) for r in rr if r['met']=='True']
        row=dict(method=m,success=line(rr),total=len(rr),sum_measured_s=sum(float(r['wall_s']) for r in rr),median_success_s=statistics.median(good) if good else None,max_peak_MiB=max(float(r['peak_mib']) for r in rr if r['peak_mib']),attempts=sum(int(r['attempts']) for r in rr),selection_contours=sum(int(r['selection_contours']) for r in rr))
        for f in ('nf','model'):
            for k in sorted({r[f] for r in rr}): row[f+'_'+k]=line([r for r in rr if r[f]==k])
        summary.append(row)
    writecsv(E/'performance_summary.csv',summary)
    lines=['# Phase 4 홀드아웃 및 비용 평가','90초·프로세스당 commit 768 MiB, 공통 수치 백엔드. 홀드아웃은 동시 작업자 최대 4개, 대표 반복은 다른 실험과 겹치지 않는 직렬 냉시작이다. wall-clock은 프로세스 시작, 선택용 contour, 재발급 contour, 발급, 독립 검증 및 파일 기록을 포함한다. 표의 합은 작업별 경과시간의 합이다. 과거 Phase 3 30초 기록은 이 표에 합치지 않았다.','|방법|목표 / 48|NF 3 / 10 / 14 (각 16)|Padé / Fresnel (각 24)|후보|선택용 contour|총 s|성공 중앙 s|peak MiB|\n|---|---:|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines[-1]+=f"\n|{s['method']}|{s['success']}|{s['nf_3']} / {s['nf_10']} / {s['nf_14']}|{s['model_Pade']} / {s['model_Fresnel']}|{s['attempts']}|{s['selection_contours']}|{s['sum_measured_s']:.1f}|{s['median_success_s']:.2f}|{s['max_peak_MiB']:.1f}|"
    for m in 'ABC':
        rr=[r for r in hold if r['method']==m]
        lines.append(f"{m} 미달 원인(분모 48) — {dict(Counter(r['reason'] for r in rr if r['met']!='True'))}.")
    for a,b in ('AB','AC','BC'):
        common=[r for r in matched if r[a+b+'_common']]
        ua=sum(float(r[a+'_wall_s']) for r in common);ub=sum(float(r[b+'_wall_s']) for r in common)
        excessive=[r for r in common if r[a+'_over_target_100x']=='True' or r[b+'_over_target_100x']=='True']
        cut=[r for r in common if r not in excessive]
        lines.append(f'{a}/{b} 공통 성공 {len(common)}건의 누적 시간 {ua:.2f}/{ub:.2f}초. 하나 이상이 두 목표보다 100배 이상 좁은 {len(excessive)}건을 별도 표시했다. 이를 제외한 {len(cut)}건에서는 '+ '/'.join(f"{sum(float(r[m+'_wall_s']) for r in cut):.2f}" for m in (a,b))+'초였다.')
    for m in 'ABC':
        rr=[r for r in repeat if r['method']==m];ts=[float(r['wall_s']) for r in rr]
        lines.append(f"직렬 반복 {m} (case {rr[0]['case']} Fresnel, 성공 {line(rr)}/5) — 평균 {statistics.mean(ts):.3f}, 중앙 {statistics.median(ts):.3f}, 범위 {min(ts):.3f}–{max(ts):.3f}초.")
    lines+=['Windows 11, AMD Ryzen 7 7800X3D, CPython 3.14.7, python-flint 0.9.0 / FLINT 3.6.0. Ubuntu/WSL의 별도 대표 검증은 수치 포함관계만 검사했다.',
    '개발 정책 비교에서 A는 동결된 기존 정책과 늘어난 시간 예산, B는 ρ=2 고차수, Cfixedbits는 가변 ρ에 1024비트 고정, C는 가변 ρ·차수·정밀도 결합 방식이다. 개발 단계에서는 이 네 가지를 A/B/C/D로 부른다. 홀드아웃 C는 개발 D다. 개발 측정은 다른 개발 작업과 동시 진행해 성능 일반화에 사용하지 않는다.']
    for m in ('A','B','Cfixedbits','C'):
        rr=[r for r in dp if r['method']==m]
        lines.append(f"개발 {m}: {line(rr)}/16, 총 {sum(float(r['wall_s']) for r in rr):.2f}초. Phase 3 원래 예산의 NF=10 성적은 0/16이었다.")
    save('phase4_experiment_report.md',lines)
    devkeys=sorted(set(map(key,grid)))
    comparisons=[]
    degrees=[]
    for k in devkeys:
        rr=[r for r in grid if key(r)==k and r['verified']=='True']
        def mindeg(seq):return min((int(r['n']) for r in seq if r['met']=='True'),default=None)
        degrees.append((k,mindeg([r for r in rr if r['rho']=='2']),mindeg(rr)))
        for n in (64,80,96,128):
            sc=[r for r in sw if key(r)==k and r.get('analytic')=='True' and int(r['n'])==n]
            base=next(r for r in sc if r['rho']=='2')
            best=min(sc,key=lambda r:F(r['delta']))
            comparisons.append(dict(case=k[0],model=k[1],n=n,Q=best['Q'],rho=best['rho'],factor=ratio(base['delta'],best['delta']),base=base['delta'],best=best['delta']))
    writecsv(E/'finite_rho_comparison.csv',comparisons)
    fixed=sum(b is not None for _,b,_ in degrees)
    opted=sum(v is not None for _,_,v in degrees)
    reduced=sum(v is not None and b is not None and v<b for _,b,v in degrees)
    best=max(comparisons,key=lambda r:r['factor'])
    small=sum(F(r['rho'])<2 for r in comparisons);large=sum(F(r['rho'])>2 for r in comparisons)
    lines=['# 수학적 결과와 해석',f"개발 NF=10에서 고정 ρ=2의 인증 차수 확대는 {fixed}/16건에 목표를 만족하는 후보를 제공했다. 가변 ρ까지 포함하면 {opted}/16건이다. 두 방식 모두 달성한 조건 중 {reduced}건에서 시험한 최소 성공 차수가 내려갔다. 전체 차수에 대한 이론적 최소는 평가하지 않았다.",
    f"64개의 (조건, 차수) 비교에서 최소 상한을 준 후보가 2보다 작은 경우 {small}건, 큰 경우 {large}건이다. 가장 큰 감소는 case {best['case']} {best['model']}, n={best['n']}, ρ={best['rho']}에서 δ {best['base']} → {best['best']} ({best['factor']:.3g}배)였다. 이 비율은 두 인증 상한의 비교이며 실제 보간 오차의 감소량은 측정하지 않았다.",
    'NF=10에서 큰 Q와 K는 복소 호 위 지수함수의 크기를 키운다. Phase 3의 64차·ρ=2에서는 16개 모두 장 나머지 δ가 절대폭 목표를 이미 초과했다. Q의 영향은 개발 표에서 같은 모델·차수별로 볼 수 있다. 개발 데이터는 NF=10뿐이다. NF 자체에 대한 최적 ρ의 단조성은 미검증이다.',
    'H1 Supported 및 H2 Supported. 유한 후보 간 엄밀한 상한 비교로 반례와 개선 사례를 얻었다. 자세한 충분조건과 교차 차수 부등식은 proofs/rho_bound_derivation.md에 있다. 개선은 실제 물리장의 변경이 아니라 동일 다항식의 나머지 예산을 줄이는 것이다.',
    f"H3 {'Supported' if summary[2]['success']>summary[0]['success'] else 'Inconclusive'} (동결 정책과 기존 A 비교). 홀드아웃 A/B/C 성공은 {summary[0]['success']}/{summary[1]['success']}/{summary[2]['success']}건이다. 고차수 B와 비교한 추가 성공 건수는 {summary[2]['success']-summary[1]['success']}건이다.",
    'H4 Inconclusive (범용 계산 효율). 동일 조건의 측정과 대표 5회 반복은 별도 비용 보고서에 있다. 모든 조건에서 contour 탐색이 상쇄된다는 증거는 없다. 한 CPU, 한 대표 반복 조건과 유한한 후보 집합만 검증했다.',
    '원본과 ρ=2의 16개 기준 구간 교집합, 원본 해시 및 실제 검증 결과는 audits/verification_results.json과 source_hash_audit.json에 기록했다. 샘플의 최대값은 인증 상한으로 사용하지 않았다. 고차수·정밀도 전체의 직적곱 대신 네 차수 전부와 n=96에서 세 정밀도의 통제 비교를 수행했다.']
    save('phase4_mathematical_report.md',lines)
    def fig(name,x,y,title,draw,log=False):
        fig,ax=plt.subplots(figsize=(7,4.2));draw(ax);ax.set(xlabel=x,ylabel=y,title=title)
        if log:ax.set_yscale('log')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(D/'figures'/name,dpi=170);plt.close(fig)
    def rho_plot(ax):
        for n in (64,80,96,128):
            rows=[r for r in sw if r['case']=='11' and r['model']=='Pade' and r['n']==str(n)]
            ax.plot([float(r['rho']) for r in rows],[float(r['epsilon']) for r in rows],'.-',label=f'n={n}')
    fig('bound_vs_rho.png','Bernstein ellipse parameter','Kernel error upper bound','Development case 11, Pade',rho_plot,True)
    def degree_plot(ax):
        s=[r for r in comparisons if r['case']=='11' and r['model']=='Pade']
        for k,label in [('base','Fixed rho=2'),('best','Best tested rho')]:ax.plot([r['n'] for r in s],[float(r[k]) for r in s],'o-',label=label)
    fig('delta_vs_degree.png','Degree','Field remainder upper bound','Development case 11, Pade',degree_plot,True)
    def successes(ax):
        for j,s in enumerate(summary):ax.bar([i+(j-1)*.24 for i in range(3)],[s['nf_'+k] for k in ('3','10','14')],width=.23,label=s['method'])
        ax.set_xticks(range(3),['3','10','14']);ax.set_ylim(0,17)
    fig('success_by_nf.png','Fresnel number','Targets met (out of 16)','Independent holdout',successes)
    def times(ax):
        for m in 'ABC':
            rr=[r for r in repeat if r['method']==m];ax.plot([int(r['rep']) for r in rr],[float(r['wall_s']) for r in rr],'o-',label=m)
    fig('runtime_comparison.png','Repetition','Cold wall time (s)','Interleaved serial measurements',times)
if __name__=='__main__':main()

