"""Post-freeze descriptive diagnostics; never feeds back to policy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import *
from collections import Counter
def main():
    grid=loadcsv(E/'development_results.csv')
    degree=[]
    for r in grid:
        if r['n']=='96' and r['rho']=='2' and r['bits']=='512':
            group=[v for v in grid if all(v[k]==r[k] for k in ('n','rho','case','model'))]
            if any(v['verified']!='True' for v in group):continue
            vals=sorted(group,key=lambda v:int(v['bits']))
            degree.append(dict(case=r['case'],model=r['model'],gram_512=r['gram_width_hi'],gram_1024=vals[-1]['gram_width_hi'],delta_512=r['delta'],delta_1024=vals[-1]['delta'],met_512=r['met'],met_1024=vals[-1]['met']))
    writecsv(E/'precision_diagnostics.csv',degree)
    groups={}
    for folder in (D/'records').glob('holdout_*_C_1'):
        scans=read(folder/'contours.json');res=read(folder/'result.json')
        for n in NS:
            best=min((r for r in scans if r.get('n')==n),key=lambda r:F(r['delta']))
            groups.setdefault((res['nf'],n),Counter())[best['rho']]+=1
    lines=['# 고정 차수 진단',f"개발 96차·ρ=2의 {len(degree)}개 정밀도 대조에서 512→1024비트로 목표 판정이 바뀐 경우 {sum(v['met_512']!=v['met_1024'] for v in degree)}건, Gram 노름 폭 상향값이 감소한 경우 {sum(F(v['gram_1024'])<F(v['gram_512']) for v in degree)}건이다. δ의 주된 항은 M과 ρ의 함수다. 비트 수는 이 항에 대한 볼 산술의 미세한 포함 여유와 Gram 폭에 영향을 준다.",
    '아래 빈도는 홀드아웃에서 각 n의 8개 후보 중 δ 상한이 가장 작은 ρ를 집계한 기술통계다. 동일 p·AR·θ 조합이 각 NF에서 반복된다. 연속 최적성이나 보편적 단조성은 주장할 수 없다.','|NF|n|최소 상한의 ρ (모델 사례 수)|\n|---|---:|---|']
    for (nf,n),v in sorted(groups.items(),key=lambda i:(int(i[0][0]),i[0][1])):
        lines[-1]+='\n|'+nf+'|'+str(n)+'|'+', '.join(k+' ('+str(v[k])+')' for k in sorted(v,key=F))+'|'
    (D/'reports/precision_and_rho_diagnostics.md').write_text('\n\n'.join(lines),encoding='utf8')
if __name__=='__main__':main()

