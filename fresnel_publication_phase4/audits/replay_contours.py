"""Independent contour replay also for development rho values never selected."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import *
from development import case
def main():
    rows=[]
    old=loadcsv(E/'rho_sweep_results.csv')
    for i in (9,10,11,12,21,22,23,24):
        for model in ('Pade','Fresnel'):
            params=case(i,model)
            archived=read(ROOT/f'fresnel_publication_phase3/runs/{i:02}_{model}_C/n64_b512.json')['certificate']['obligations']
            with ctx.workprec(512):
                Q=verify.point(archived['geometry']['evidence']['Q'])
                th=A(params['theta'])
                K=2*A.pi()*(10/th)/(10*th/A(params['nf']))
                for rho in RHOS:
                    margins,c=gen.contour(K,Q,model,rho)
                    M=verify.contour_check(dict(analyticity=margins,contour=c),Q,K,model,rho)
                    subset=[r for r in old if r['case']==str(i) and r['model']==model and r['rho']==rho]
                    if len(subset)!=4 or any(r['M']!=dec(dy(gen.dyadic(M))) for r in subset):raise ValueError((i,model,rho))
                    rows.append(dict(case=i,model=model,rho=rho,panels=len(c['arc_uppers']),verified=True))
    writecsv(D/'audits/contour_replay.csv',rows)
if __name__=='__main__':main()

