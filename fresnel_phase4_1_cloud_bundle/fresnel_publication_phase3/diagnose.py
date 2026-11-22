"""Diagnose the exact failure budget for the selected verified B candidates."""
import csv, json
from collections import Counter
from fractions import Fraction
from experiment import D, stats, pilot

def append_diagnosis(results):
    failures=[r for r in results if r['method']=='B' and r['status']!='target_met' and r.get('degree')]
    rows=[]
    for r in failures:
        path=D/'runs'/f"{r['case']:02}_{r['model']}_B"/f"n{r['degree']}_b{r['bits']}.json"
        cert=json.loads(path.read_text())['certificate']
        g,slack,M,e=stats(cert)
        l,u,w,rel,rok,aok=pilot.metric(cert,Fraction('.01'),Fraction('.00001'))
        # The interval contains at least the upper-side slack even if clipped.
        # Positive lower implies width >= 2*slack; u+l <= 2*u.
        forces_abs=slack>Fraction('1e-5')
        forces_rel=(u>0 and 2*slack>Fraction('.01')*u)
        # If l=0, rel already fails. The test above also excludes success
        # for any positive lower at the observed upper.
        rows.append((r['case'],r['model'],forces_abs,forces_rel,float(slack),float(g),l==0))
    with (D/'failure_analysis.md').open('a',encoding='utf8') as f:
        print('\n## 최선 후보의 나머지 예산',file=f)
        print('\n아래 절대폭 불가능 판정은 δ > 1e-5이다. 상대폭 불가능 판정은 2δ > 0.01U이다. 양의 하한에서는 폭 ≥ 2δ이고 상대폭 ≥ 폭/U이며, 하한 0은 규칙상 미달이다. 두 판정은 충분조건이다.',file=f)
        print('\n|조건|모델|δ만으로 절대폭 미달|현재 U에서 δ만으로 상대폭 미달|장 δ|Gram 제곱 폭|하한 0|\n|---|---|---|---|---:|---:|---|',file=f)
        for i,m,a,b,d,g,z in rows: print(f'|{i}|{m}|{a}|{b}|{d:.4g}|{g:.4g}|{z}|',file=f)
