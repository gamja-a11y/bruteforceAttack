import time
from common import *
import rho_certifier as gen
import rho_independent_verifier as verify
from flint import arb as A, acb as C,ctx
sys.path.insert(0,str(ROOT/'fresnel_publication_phase3_audit_repaired'))
from repaired_analyze import gram_width
sys.path.insert(0,str(ROOT/'fresnel_publication_phase2'))
from adaptive_certification import peak_mib
RHOS=['1.1','1.2','1.3','1.5','1.75','2','2.5','3']
NS=[64,80,96,128]
BITS=[512,768,1024]
def evaluate(params,n,b,rho,path):
    start=time.perf_counter()
    expected=dict(**params,degree=n,bits=b,rho=rho,Q=None)
    rec=gen.certify(**expected)
    ev={k:v['evidence'] for k,v in rec['obligations'].items()}
    l,u=[dy(ev['endpoints'][k]) for k in ('lower','upper')]
    w=u-l
    rel=2*w/(u+l) if u+l else None
    rec['target_met']=w<=F('1e-5') and l>0 and rel<=F('.01')
    issue=time.perf_counter()-start
    proof=verify.check(rec,expected)
    dump(path,dict(expected=expected,certificate=rec,verification=proof))
    gl,gu=map(dy,ev['gram']['G'])
    gwlo,gwhi=gram_width(gl,gu)
    return dict(n=n,bits=b,rho=rho,verified=True,met=proof['target_met'],L=dec(l,False),U=dec(u),width=dec(w),relative=dec(rel) if rel is not None else '',M=dec(dy(ev['contour']['M'])),epsilon=dec(dy(ev['interpolation']['epsilon'])),delta=dec(dy(ev['remainder']['delta'])),gram_squared_width=dec(gu-gl),gram_width_lo=dec(gwlo,False),gram_width_hi=dec(gwhi),issue_s=issue,elapsed_s=time.perf_counter()-start,peak_mib=peak_mib(),file=path.relative_to(D).as_posix())
def scan(params,bits,rhos):
    rows=[]
    with ctx.workprec(bits):
        ax,ay,z,lam,K,qm,Q=gen.geometry(*(A(params[k]) for k in ('p','nf','theta','ar','radius')),None)
        for rho in rhos:
            start=time.perf_counter()
            try:
                _,cont=gen.contour(K,Q,params['model'],rho)
                vals=list(map(dy,cont['arc_uppers']))
                j=vals.index(max(vals))
                r=A(rho)
                t=A.pi()*(2*j+1)/512
                point=C(Q/2+Q*(r+1/r)*t.cos()/4,Q*(r-1/r)*t.sin()/4)
                sample=gen.residual(point,K,params['model']).abs_lower()
                M=A((int(cont['M'][0]),cont['M'][1]))
                factor=dec(dy(gen.dyadic((M/sample).upper()))) if sample>0 else None
                for n in NS:
                    eps=(4*M/(r**n*(r-1))).upper()
                    d=(A.pi()*A(params['nf'])*eps).upper()
                    rows.append(dict(rho=rho,n=n,bits=bits,analytic=True,Q=dec(dy(gen.dyadic(Q))),M=dec(max(vals)),epsilon=dec(dy(gen.dyadic(eps))),delta=dec(dy(gen.dyadic(d))),inflation_upper=factor,contour_s=time.perf_counter()-start))
            except ValueError as e: rows.append(dict(rho=rho,analytic=False,reason=str(e),bits=bits,contour_s=time.perf_counter()-start))
    return rows
if __name__=='__main__':
    params=dict(p='4',nf='10',ar='1',theta='0.1',radius='5',model='Pade')
    print(evaluate(params,64,512,'1.75',D/'records/probe.json'))

