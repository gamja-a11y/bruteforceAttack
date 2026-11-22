"""Certified positive GL probe and a tightened continuous geometric bound."""
import argparse
import csv
import json
from math import comb
from pathlib import Path
from time import perf_counter

from scipy.special import roots_legendre
from fresnel_certified_bound import backend
from fresnel_certified_bound_v2 import geometric, rule
from fresnel_error_bound_validation import symbolic_proof


def legendre(A,n,x):
    a,b=A(1),x
    for j in range(2,n+1):
        a,b=b,((2*j-1)*x*b-(j-1)*a)/j
    return b,a


def gauss(A,n):
    nodes,weights=[],[]
    for approximate in roots_legendre(n)[0]:
        mid=A(float(approximate))
        h=A(1)/2**40
        lo,hi=mid-h,mid+h
        l,r=legendre(A,n,lo)[0],legendre(A,n,hi)[0]
        assert l*r<0 and (not nodes or nodes[-1].upper()<lo)
        for _ in range(45):
            center=(lo+hi)/2
            pcenter=legendre(A,n,center)[0]
            if pcenter*l>0:
                lo,l=center,pcenter
            elif pcenter*r>0:
                hi,r=center,pcenter
            else:
                break
        x=lo.union(hi)
        pn,previous=legendre(A,n,x)
        d=n*(previous-x*pn)/(1-x*x)
        w=1/((1-x*x)*d*d)  # [-1,1] weight divided by two
        assert w>0
        nodes.append(x)
        weights.append(w)
    assert sum(weights).contains(1)
    return [(x+1)/2 for x in nodes],weights


def physics(A):
    p=A(12)
    pi=A.pi()
    ax=(25*pi*(1+2/p).gamma()/(1+1/p).gamma()**2).sqrt()
    z,lam=A(100)/3,A(3)/10
    return pi,ax,z,lam,2*pi*z/lam


def continuous(A):
    pi,a,z,lam,K=physics(A)
    Q=A(1)/2
    assert ((a*A(2).sqrt()+5)/z)**2<Q
    coeff={}
    base={}
    # Expand (x²+y²+x'²+y'²-2xx'-2yy')^n by binomial parity
    # in the separated x/y powers. Integrate all retained terms exactly.
    power=(1+1/K**2).sqrt()*K/512
    amplitude=7/(1536*K)
    d=(1-1/((1+Q)*(1+3*Q/4)))/Q
    def mu(i,j):
        return a**(2*i+2*j+2)/36*((A(2*i+1)/12).gamma())*((A(2*j+1)/12).gamma())/(
            1+A(2*i+2*j+2)/12).gamma()
    assert mu(0,0).overlaps(100*pi)
    for n,scale in ((5,power+amplitude),(6,-power*d)):
        for k in range(n+1):
            for i in range(k+1):
                for j in range(n-k+1):
                    key=(k-i,n-k-j)
                    v=comb(n,k)*comb(2*k,2*i)*comb(2*(n-k),2*j)*mu(i,j)/z**(2*n)
                    coeff[key]=coeff.get(key,A(0))+v*scale
                    if n==5:
                        base[key]=base.get(key,A(0))+v*scale
    disk={}
    def norm(c):
        s=A(0)
        for (i,j),v in c.items():
            for (k,l),w in c.items():
                x,y=i+k,j+l
                if (x,y) not in disk:
                    u,t=A(x)+A(1)/2,A(y)+A(1)/2
                    disk[x,y]=25**(x+y)*u.gamma()*t.gamma()/((u+t).gamma()*pi*(x+y+1))
                s+=v*w*disk[x,y]
        return s.sqrt()/(lam*z)
    assert norm({(0,0):A(1)}).overlaps(1/(lam*z))
    v2,_=geometric(A)
    original=norm(base)
    tightened=norm(coeff)
    assert original.overlaps(v2) and tightened<original
    return original,tightened


def probe(A,C,n,panels,kind):
    _,a,z,lam,K=physics(A)
    if kind=="GL":
        nodes,weights=gauss(A,n)
        S=A(1)  # exact because the verified GL weights are positive
        exact=2*n-1
    else:
        xx,ww,ss=rule(n)
        rat=lambda x:A(x.numerator)/x.denominator
        nodes,weights=list(map(rat,xx)),list(map(rat,ww))
        S,exact=rat(ss),n
    J=C(0,1)

    def box(i):
        h=A(1)/(2*panels)
        return C((2*i+1)*h+A(0,1)*2*h,A(0,1)*2*h)

    def geom(t):
        b=1+t**12
        if not b.real>0:
            raise ArithmeticError("sector branch enclosure")
        e=b**(-A(1)/12)
        d=-e*t**11/b
        return a*e,a*t*e,a*d,a*(e+t*d)

    def integrand(g,u,swap,sx,sy):
        X,Y,dX,dY=g
        if swap:
            X,Y,dX,dY=Y,X,dY,dX
        dx,dy=sx*X-3,sy*Y-4
        q=u*(dx*dx+dy*dy)/z**2
        b,c,d=1+3*q/4+q*q/16,16+12*q+q*q,48+84*q+35*q*q
        if any(not v.real>0 for v in (1+q,b,c,d)):
            raise ArithmeticError("kernel holomorphy enclosure")
        t=(1+q).sqrt()
        g0=(q/(1+t))**5
        h=J*K*g0/c
        part=(J*K*(q/2+q*q/4)/b).exp()*(J*g0*(t*t+5*t+1)/(K*t**3*d)
              +(1/(1+q)+J/(K*t**3))*(h+h*h/2))
        return part*(dx*sy*dY-dy*sx*dX)*(sx*sy*(-1 if swap else 1))/2

    maxima,total=A(0),C(0)
    for i in range(panels):
        gb=geom(box(i))
        for j in range(panels):
            ub=box(j)
            for swap in (False,True):
                for sx in (-1,1):
                    for sy in (-1,1):
                        maxima+=integrand(gb,ub,swap,sx,sy).abs_upper()
    # Two-parameter telescoping error, including negative-weight amplification.
    remainder=(maxima*(1+S)**2/(2**exact*panels**2*lam*z)).upper()
    if kind=="GL":
        for i in range(panels):
            gs=[geom(C((i+t)/panels)) for t in nodes]
            for j in range(panels):
                us=[C((j+u)/panels) for u in nodes]
                for swap in (False,True):
                    for sx in (-1,1):
                        for sy in (-1,1):
                            total+=sum(w*v*integrand(g,u,swap,sx,sy)
                                       for g,w in zip(gs,weights) for u,v in zip(us,weights))
        total/=panels**2*J*lam*z
    inflation=A(0,1)*remainder
    field=(str(total.real+inflation),str(total.imag+inflation)) if kind=="GL" else (None,None)
    return dict(M_sum_over_panels_and_sectors=str(maxima),S=str(S),exact_degree=exact,
                interval_error=str(remainder),error_display=float(remainder),
                real_enclosure=field[0],imag_enclosure=field[1])


PROOF="""# v3 proof and scope

**Proven.** The exact Padé identities and their signs are checked algebraically.
For real q>=0, |A(q)|<=sqrt(1+K^-2)/(1+q). Also
delta/q^5=1/[(sqrt(1+q)+1)^5(16+12q+q^2)]<=1/[512(1+3q/4)].
The product reciprocal f(q)=1/[(1+q)(1+3q/4)] is convex on q>=0.
Its chord on [0,Q] is 1-dq and bounds it above. Therefore the
pointwise residual is bounded by (C q^5 - sqrt(1+K^-2) K d q^6/512),
a nonnegative polynomial on the physical domain. Its continuous field L2
bound uses exact superellipse gamma moments and disk beta moments of the
squared polynomial, with outward rounding. The disk monomial moment is
R^(2i+2j) B(i+1/2,j+1/2)/(pi(i+j+1)). The normalized constant moment is one.
Both geometric bounds apply to the same unfit raw absolute field norm.

**Proven.** The boundary sweep uses eight reflected/swapped sectors. Their
orientation determinants multiply n ds=(dy,-dx). The q=0-safe substitution
H(q)=1/2 integral_0^1 G(uq)du preserves analyticity and gives H(0)=0.
The implementation uses the Padé residual, not a Fresnel phase approximation.
At each real cell the Cauchy contour is contained in a rectangular Arb box
around the complex disk with twice the real half-width. All branch/pole
conditions are checked on that box. A degree-d Taylor tail is <=M*2^-d.
On [0,1], a rule of degree d and absolute-weight sum S has error <=
(1+S) M*2^-d. Telescoping over both coordinates gives the factor (1+S)^2.

**Interval-certified.** SciPy supplies candidate locations only. Disjoint
rational root brackets with Arb-certified sign changes account for all n
Legendre roots. The exact Gauss weight formula on each bracket certifies
positive weights. Complex contributions are summed before adding remainder
to each component separately. No experimental convergence is used as a bound.

**Unverified.** The high-order boundary probe is at (3,4); continuous ROI
phase-preserving extension, a Taylor-model fallback and automatic tolerance-based
selection are not implemented. The continuous improvement is geometric.
The empirical oscillatory reference is only a comparator.
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--high-order-only",action="store_true")
    parser.add_argument("--out",type=Path,default=Path("certified_bound_v3_results"))
    args=parser.parse_args()
    settings=(("NC",12,16),("NC",12,24),("NC",16,24),("GL",8,16),
              ("GL",8,24),("GL",12,24),("GL",16,24),("GL",24,24))
    if args.high_order_only:
        settings=settings[-1:]
    print("Continuous polynomial certificates at 64/128/192/256 bits; boundary probes:",settings,flush=True)
    if args.plan:
        return
    flint,A,C,ctx=backend()
    args.out.mkdir(parents=True,exist_ok=False)
    proof=symbolic_proof()
    budgets=[]
    for bits in (64,128,192,256):
        ctx.prec=bits
        tick=perf_counter()
        original,tight=continuous(A)
        budgets.append(dict(bits=bits,v2=str(original.upper()),v3=str(tight.upper()),
                            v3_display=float(tight.upper()),seconds=perf_counter()-tick,
                            status="interval-certified"))
    ctx.prec=128
    rows=[]
    for kind,n,panels in settings:
        start=perf_counter()
        row=dict(method=kind,degree_or_nodes=n,panels=panels,real_half_width=1/(2*panels))
        try:
            row.update(probe(A,C,n,panels,kind),status="interval-certified",
                       scope="pointwise field" if kind=="GL" else "remainder only")
        except (ArithmeticError,AssertionError) as exc:
            row.update(status="unverified",reason=str(exc) or type(exc).__name__)
        rows.append(dict(**row,seconds=perf_counter()-start))
        print(kind,n,panels,row["status"],flush=True)
    report=dict(backend=flint.__version__,proof=proof,continuous_geometry=budgets,boundary=rows,
                empirical_oscillatory_reference=4.678284775e-6,
                continuous_phase_preserving_status="unverified",adaptive_status="unverified",
                reproduce="python fresnel_certified_bound_v3.py --out NEW_DIRECTORY")
    (args.out/"certificate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"proof.md").write_text(PROOF,encoding="utf-8")
    for name,data in (("boundary_convergence",rows),("continuous_budget",budgets)):
        with (args.out/f"{name}.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            writer.writeheader()
            writer.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(layout="constrained")
    for kind in ("NC","GL"):
        subset=[r for r in rows if r["method"]==kind and "error_display" in r]
        ax.semilogy([f"{r['degree_or_nodes']}/{r['panels']}" for r in subset],
                    [r["error_display"] for r in subset],"o-",label=kind)
    ax.set(xlabel="Degree or node count / panels per axis",ylabel="Certified pointwise remainder")
    ax.legend()
    fig.savefig(args.out/"boundary_convergence.png",dpi=170)


if __name__=="__main__":
    main()
