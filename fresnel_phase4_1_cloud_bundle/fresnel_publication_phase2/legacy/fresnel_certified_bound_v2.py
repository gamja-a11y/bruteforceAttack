"""Independent geometric ROI certificate and high-order boundary probe audit."""
import argparse
import csv
import json
from fractions import Fraction
from functools import lru_cache
from math import comb
from pathlib import Path
from time import perf_counter

from fresnel_certified_bound import backend, verify
from fresnel_error_bound_validation import symbolic_proof


@lru_cache(None)
def rule(n):
    """Closed interpolatory rule on [0,1], exact through degree n."""
    xs = tuple(Fraction(i,n) for i in range(n+1))
    weights = []
    for i,x in enumerate(xs):
        c = [Fraction(1)]
        for y in xs[:i]+xs[i+1:]:
            b = [Fraction(0)]*(len(c)+1)
            for j,v in enumerate(c):
                b[j] -= v*y/(x-y)
                b[j+1] += v/(x-y)
            c=b
        weights.append(sum(v/(j+1) for j,v in enumerate(c)))
    assert all(sum(w*x**j for x,w in zip(xs,weights))==Fraction(1,j+1) for j in range(n+1))
    return xs,weights,sum(map(abs,weights))


def parameters(A):
    pi=A.pi()
    z,lam=A(100)/3,A(3)/10
    a=(100*pi*(1+A(1)/6).gamma()/(4*(1+A(1)/12).gamma()**2)).sqrt()
    return pi,z,lam,a,2*pi*z/lam


def geometric(A):
    pi,z,lam,a,k=parameters(A)
    def coeff(n):
        c={}
        for l in range(n+1):
            for i in range(l+1):
                for j in range(n-l+1):
                    mu=(4*a**(2*(i+j)+2)/144*((A(2*i+1)/12).gamma())
                        *((A(2*j+1)/12).gamma())/(1+A(2*i+2*j+2)/12).gamma())
                    key=l-i,n-l-j
                    c[key]=c.get(key,A(0))+comb(n,l)*comb(2*l,2*i)*comb(2*(n-l),2*j)*mu/z**(2*n)
        return c
    def disk(i,j):
        u,v=A(i)+A(1)/2,A(j)+A(1)/2
        return 5**(2*(i+j))*u.gamma()*v.gamma()/((u+v).gamma()*pi*(i+j+1))
    memo={}
    def norm(n):
        c=coeff(n)
        result=A(0)
        for (i,j),u in c.items():
            for (k,l),v in c.items():
                key=i+k,j+l
                if key not in memo:
                    memo[key]=disk(*key)
                result+=u*v*memo[key]
        return result.sqrt()
    return ((7/(1536*k)+k*(1+1/k**2).sqrt()/512)*norm(5)/(lam*z),
            (1+1/k**2).sqrt()*(k/512)**3*norm(15)/(6*lam*z))


def boundary(A,C,x,y,n,panels):
    _,z,lam,a,k=parameters(A)
    j=C(0,1)
    def rational(v):
        return A(v.numerator)/v.denominator
    xs,ws,s=rule(n)
    nodes,weights=list(map(rational,xs)),list(map(rational,ws))
    s=rational(s)
    # Unit-interval parameter boxes and their radius-2h Cauchy disks.
    def box(i,size,wide):
        mid=A(2*i+1)/(2*size)
        r=A(1)/(size*(1 if wide else 2))
        return C(mid+A(0,1)*r,A(0,1)*r) if wide else mid

    def location(t):
        b=1+t**12
        if not b.real>0:
            raise ArithmeticError("sector branch cut")
        e=b**(-A(1)/12)
        d=-e*t**11/b
        return a*e,a*t*e,a*d,a*(e+t*d)

    def f(g,u,swap,sx,sy):
        X,Y,dX,dY=g
        if swap:
            X,Y,dX,dY=Y,X,dY,dX
        dx,dy=sx*X-x,sy*Y-y
        q=u*(dx*dx+dy*dy)/z**2
        if not (1+q).real>0:
            raise ArithmeticError("kernel branch cut")
        t=(1+q).sqrt()
        b,c,d=1+3*q/4+q*q/16,16+12*q+q*q,48+84*q+35*q*q
        if any(not v.real>0 for v in (b,c,d)):
            raise ArithmeticError("kernel pole enclosure")
        phi=(q/2+q*q/4)/b
        s0=(q/(t+1))**5
        gap=j*k*s0/c
        G=(j*k*phi).exp()*(j*s0*(t*t+5*t+1)/(k*t**3*d)
            +(1/(1+q)+j/(k*t**3))*(gap+gap*gap/2))
        return G*(dx*sy*dY-dy*sx*dX)*(sx*sy*(-1 if swap else 1))/2

    total,err=C(0),A(0)
    for it in range(panels):
        tbox=location(box(it,panels,True))
        tnodes=[location(C((it+t)/panels)) for t in nodes]
        for iu in range(panels):
            ubox=box(iu,panels,True)
            unodes=[C((iu+u)/panels) for u in nodes]
            for swap in (False,True):
                for sx in (-1,1):
                    for sy in (-1,1):
                        bound=f(tbox,ubox,swap,sx,sy).abs_upper()
                        err+=bound*(1+s)**2/2**n
                        total+=sum(w*v*f(g,u,swap,sx,sy) for g,w in zip(tnodes,weights)
                                   for u,v in zip(unodes,weights))
    error=err/(panels**2*lam*z)
    v=total/(panels**2*j*lam*z)
    component=A(0,1)*error
    return C(v.real+component,v.imag+component),error.upper()


AUDIT = """# Mathematical audit

- **Proven, kernel and Taylor.** Cross-multiplied Padé identities and denominator signs
  are checked symbolically. For real phase error x, the integral Taylor
  remainder of exp(ix) is at most |x|^3/6 because |exp(itx)|=1.
- **Proven, normal and continuity.** With q=|r'-r|²/z², div((r'-r)H)=2H+2qH'=G.
  For CCW orientation n ds=(dy,-dx). Every reflected or swapped sector is
  multiplied by its orientation determinant. Swapping x,y is an ROI symmetry
  only at AR=1. H(q)=1/2 integral_0^1 G(uq)du removes division by zero,
  gives H(0)=G(0)/2=0, and preserves analyticity. G=O(q^5) there.
- **Proven, v1 interpolation.** The Lobatto-series coefficient convention and
  affine antiderivative are consistent. A rectangle covering the complete
  Bernstein ellipse must avoid the square-root cut and all denominator zeros.
  Coefficient balls include rounding. However, use of q.mid() after an
  outward-rounded intersection does not explicitly prove the center remains in
  the exact piecewise support in edge cases. v2 bypasses that argument; it does
  not reissue the v1 composite certificate.
- **Proven, v1 enclosure design.** Pointwise interval bounds and positive
  normalized disk cell weights bound the continuous L2 integral; enclosing
  terms independently loses cancellation. A plotted decimal is not the
  certificate; use the outward endpoint of the stored ball.
- **Proven, v2 high-order remainder.** On each two-parameter rectangle, check
  holomorphy and |f|<=M on a complex polydisk with radii twice the real
  half-widths. Cauchy's estimate bounds the degree-n Taylor tail in either
  parameter by M*2^-n. The exact-rational interpolatory rule has weight
  absolute sum S. Telescoping the two one-axis errors gives
  M*(1+S)^2*2^-n times rectangle area. The complex quadrature sum is accumulated
  before adding the enclosure. This is a certified pointwise test only.
- **Proven, continuous geometric norm.** J5 and J15 are polynomials in x²,y²
  with positive coefficients from gamma moments. Square them algebraically
  and integrate each term using the exact normalized disk beta moment.
  The weights integrate the physical radius-5 disk divided by 25*pi.
  Both B0 and an independently valid composite B_cert bound the same raw
  ||U_RS-U_PP|| with prefactor 1/(i lambda z). Taking the minimum of two
  independently certified upper endpoints is valid; no numerical phase fit
  or relative RS normalization is used.
- **Unverified.** No phase-preserving continuous-ROI certificate from the
  high-order probes. The empirical B0_osc2 is a benchmark only. Interval GL
  would reduce node cost but requires verified nodes and weights; derivative
  Taylor models and analytic subdivision near stationary phase are alternatives.
  Oscillatory rules require a separate validated remainder.
"""


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",action="store_true")
    p.add_argument("--out",type=Path,default=Path("certified_bound_v2_results"))
    args=p.parse_args()
    print("Arb: exact disk moments; two certified boundary budgets; ROI comparison against v1.")
    if args.plan:
        return
    flint,A,C,ctx=backend()
    args.out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    proof=symbolic_proof()
    rows=[]
    for bits in (64,128,192):
        ctx.prec=bits
        t=perf_counter()
        b0,rho=geometric(A)
        rows.append(dict(test="continuous_geometry",bits=bits,B0=str(b0.upper()),
                         rho=str(rho.upper()),seconds=perf_counter()-t,status="interval-certified"))
    for n,panels in ((8,8),(12,16),(12,24),(16,24)):
        t=perf_counter()
        try:
            v,error=boundary(A,C,A(3),A(4),n,panels)
            row=dict(test="boundary_probe_3_4",degree=n,panels=panels,bits=192,
                     field=str(v),error=str(error),error_display=float(error),
                     status="interval-certified")
        except ArithmeticError as exc:
            row=dict(test="boundary_probe_3_4",degree=n,panels=panels,bits=192,
                     reason=str(exc),status="unverified")
        rows.append({**row,"seconds":perf_counter()-t})
        print(rows[-1]["test"],n,panels,rows[-1]["status"],flush=True)
    for roi in (2,4,8):
        exact,display,*_,time=verify(32,32,16,roi,128,2,("12","10","0.3","1"))
        rows.append(dict(test="v1_ROI_regression",roi=roi,B0=exact["B_cert"],
                         error_display=display["boundary"],seconds=time["total_seconds"],
                         status="empirical",reason="Inherited enclosure; not reissued as a v2 certificate"))
    report={"proof":proof,"flint":flint.__version__,"elapsed":perf_counter()-start,
            "continuous_geometry_B0_ball":str(b0.upper()),"rho3_ball":str(rho.upper()),
            "phase_preserving_continuous_status":"unverified",
            "matched_accuracy_wall_clock_status":"unverified: distinct tasks; no speedup inferred",
            "empirical_reference":4.678284775e-6,"records":rows,
            "reproduce":"python fresnel_certified_bound_v2.py --out NEW_DIRECTORY"}
    (args.out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"proof.md").write_text(AUDIT,encoding="utf-8")
    with (args.out/"convergence.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,group,title in zip(axes,("boundary_probe_3_4","v1_ROI_regression"),
                              ("Certified pointwise remainder","v1 ROI regression (empirical)")):
        items=[r for r in rows if r["test"]==group and "error_display" in r]
        ax.semilogy([f"{r['degree']}/{r['panels']}" if "degree" in r else str(r["roi"])
                     for r in items],[r["error_display"] for r in items],"o-")
        ax.set(title=title,xlabel="Degree / panels" if group.startswith("boundary") else "Cells per ROI axis")
    fig.savefig(args.out/"convergence.png",dpi=160)


if __name__=="__main__":
    main()
