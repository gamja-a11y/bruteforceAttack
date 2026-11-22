"""Direct continuous residual L2 certificate using global spectral moments."""
import argparse
import csv
import json
from math import comb
from pathlib import Path
from time import perf_counter

from fresnel_certified_bound import backend
from fresnel_certified_bound_v3 import continuous
from fresnel_error_bound_validation import symbolic_proof


def kernel(q,k):
    from flint import acb
    i=acb(0,1)
    b,c,d=1+3*q/4+q*q/16,16+12*q+q*q,48+84*q+35*q*q
    if any(not v.real>0 for v in (1+q,b,c,d)):
        raise ArithmeticError("Unverified square-root branch or denominator")
    t=(1+q).sqrt()
    s=(q/(1+t))**5
    a=1/(1+q)+i/(k*t**3)
    da=i*s*(t*t+5*t+1)/(k*t**3*d)
    return (i*k*(q/2+q*q/4)/b).exp()*(da+a*((i*k*s/c).exp()-1))


def physical(A,p,nf,theta):
    pi=A.pi()
    a=(25*pi*(1+2/p).gamma()/(1+1/p).gamma()**2).sqrt()
    z,lam=10/theta,10*theta/nf
    r=a*2**(A(1)/2-1/p)
    qmax=((r+5)/z)**2
    Q=A(1)/1024
    while not qmax<Q:
        Q+=A(1)/1024
    return pi,a,z,lam,2*pi*z/lam,Q


def interpolation(A,C,P,k,Q,n):
    # rho=2, ellipse semiaxes 5/4,3/4 on [-1,1]. First prove
    # analyticity on a rectangle containing its full closed interior.
    x,y=Q/2, Q/2
    rectangle=C(x+A(0,1)*y*5/4,A(0,1)*y*3/4)
    kernel(rectangle,k)
    M=A(0)
    for j in range(512):
        angle=(2*A.pi()*j/512).union(2*A.pi()*(j+1)/512)
        q=C(x+y*5/4*angle.cos(),y*3/4*angle.sin())
        M=M.union(kernel(q,k).abs_upper()).upper()
    error=(4*M/2**n).upper()
    nodes=[(A.pi()*i/n).cos() for i in range(n+1)]
    vals=[kernel(C(Q*(1+t)/2),k) for t in nodes]
    c=[sum(v*(A.pi()*j*i/n).cos()/(2 if i in (0,n) else 1)
           for i,v in enumerate(vals))*2/n for j in range(n+1)]
    c[0]/=2
    c[-1]/=2
    prev,cur=P([1]),P([-1,2])
    poly=P([c[0]])+cur*c[1]
    for j in range(2,n+1):
        prev,cur=cur,2*P([-1,2])*cur-prev
        poly+=cur*c[j]
    return poly.coeffs(),error,M


def aperture_transform(A,C,coeff,p,a,z,Q):
    """Coefficients of the integrated polynomial in (x/5)^2,(y/5)^2."""
    n=len(coeff)-1
    g=[(A(2*j+1)/p).gamma() for j in range(n+1)]
    d=[(1+A(2*j+2)/p).gamma() for j in range(n+1)]
    ap=[(a*a/(z*z*Q))**j for j in range(n+1)]
    wp=[(25/(z*z*Q))**j for j in range(n+1)]
    base=4*a*a/p**2
    mu=[[base*ap[i+j]*g[i]*g[j]/d[i+j] for j in range(n-i+1)] for i in range(n+1)]
    binom=[[comb(2*k,2*i) for i in range(k+1)] for k in range(n+1)]
    radial=[[coeff[k+l]*comb(k+l,k) for l in range(n-k+1)] for k in range(n+1)]
    result={}
    for i in range(n+1):
        for j in range(n-i+1):
            s=C(0)
            for k in range(i,n-j+1):
                for l in range(j,n-k+1):
                    s+=radial[k][l]*mu[k-i][l-j]*(binom[k][i]*binom[l][j])
            result[i,j]=s*wp[i+j]
    return result


def disk_gram(A,R,coeff):
    n=max(i+j for i,j in coeff)
    stride=2*n+1
    re=[A(0)]*(stride*n+n+1)
    im=re.copy()
    for (i,j),c in coeff.items():
        re[i*stride+j],im[i*stride+j]=c.real,c.imag
    r,s=R(re),R(im)
    square=(r*r+s*s).coeffs()
    rising=[A(1)]
    fac=[A(1)]
    for i in range(1,2*n+2):
        rising.append(rising[-1]*(A(i)-A(1)/2))
        fac.append(fac[-1]*i)
    gram=A(0)
    for index,value in enumerate(square):
        i,j=divmod(index,stride)
        if i+j>2*n:
            assert value.is_zero()
            continue
        gram+=value*rising[i]*rising[j]/fac[i+j+1]
    assert gram.is_finite() and gram.upper()>=0
    return gram


def certify(condition,n,bits):
    from flint import acb_poly,arb_poly
    _,A,C,ctx=backend()
    ctx.prec=bits
    p,nf,theta=map(A,condition)
    pi,a,z,lam,k,Q=physical(A,p,nf,theta)
    t=perf_counter()
    coeff,eps,M=interpolation(A,C,acb_poly,k,Q,n)
    a_time=perf_counter()-t
    t=perf_counter()
    spatial=aperture_transform(A,C,coeff,p,a,z,Q)
    source_time=perf_counter()-t
    t=perf_counter()
    gram=disk_gram(A,arb_poly,spatial)/(lam*z)**2
    upper=gram.upper().sqrt().upper()
    lower=gram.lower().sqrt().lower() if gram.lower()>0 else A(0)
    e=(100*pi*eps/(lam*z)).upper()
    final=(upper+e).upper()
    final_low=(lower-e).lower() if lower>e else A(0)
    return dict(p=condition[0],nf=condition[1],theta=condition[2],degree=n,bits=bits,
                q_upper=str(Q),ellipse_maximum=str(M),kernel_uniform_error=str(eps),
                polynomial_L2_squared=str(gram),field_approximation_error=str(e),
                upper=str(final),lower=str(final_low),L2_squared_upper=str((final**2).upper()),
                upper_display=float(final),lower_display=float(final_low),status="interval-certified",
                interpolation_seconds=a_time,aperture_seconds=source_time,gram_seconds=perf_counter()-t)


def empirical_small():
    import mpmath as mp
    # Circular p=2 test: radial ROI and aperture variables, with one aperture angle.
    results=[]
    with mp.workdps(45):
        z,lam=mp.mpf(200),mp.mpf(5)/3
        k=2*mp.pi*z/lam
        for order in (8,12):
            xx, ww=mp.gauss_quadrature(order,"legendre")
            rule=[((x+1)/2,w/2) for x,w in zip(xx,ww)]
            norm=mp.mpf(0)
            for r,wr in rule:
                field=mp.mpc(0)
                for s,ws in rule:
                    for u,wu in rule:
                        q=(25*r*r+100*s*s-100*r*s*mp.cos(mp.pi*u))/z**2
                        t=mp.sqrt(1+q)
                        d=(q/(1+t))**5
                        a=1/(1+q)+1j/(k*t**3)
                        da=1j*d*(t*t+5*t+1)/(k*t**3*(48+84*q+35*q*q))
                        phase=(q/2+q*q/4)/(1+3*q/4+q*q/16)
                        g=mp.exp(1j*k*phase)*(da+a*mp.expm1(1j*k*d/(16+12*q+q*q)))
                        field+=g*ws*wu*s
                norm+=2*r*wr*abs(field*200*mp.pi/(lam*z))**2
            results.append(dict(order=order,digits=45,L2=mp.nstr(mp.sqrt(norm),35),
                                status="empirical"))
    return results


PROOF="""# v6 direct continuous L2 certificate

## Audit of v5
The quadratic Gram expression correctly uses the real part of c_i conj(c_j)
and a factor of two off the diagonal for real monomials. The polar antiderivative
includes the radial Jacobian; 8/(25*pi) normalizes the octant for AR=1.
For the Cauchy polydisk GL estimate the Taylor tail is 2^-(2n-1) M per variable,
and two telescoped quadratures with positive weights give the factor four.
The directional third derivative is G'''(q')³+3G''q'q'', with q'''=0.
The real interval derivative majorants and radial-plus-arc displacement
produce a valid but loose remainder. v6 does not reuse that remainder or
the v5 serialized center-field replay as inputs to any certificate.

## Factorization and symmetry (proven)
Let D denote the exact complex RS-I minus Padé kernel including 1/(i lambda z).
By Fubini the squared norm is int_ROI int_A int_A D(r,s) conj(D(r,t))
dA(s)dA(t)dA(r)/(25*pi). A modulus or triangle inequality inside either
source integral discards the cross-term cancellation. Hermitian exchange
permits a real-part integral on half the paired domain; reflections of source
and observation coordinates together permit symmetry reduction. Neither
reduces the number of independent continuous coordinates for generic p=12.
For the circular p=2 case, continuous rotation reduces the observation
dependence to radius; this is used only in the empirical direct comparison.

## Interval-certified polynomial method
The exact whole residual is used, with no Taylor phase truncation. Complex
Arb boxes check analyticity on the entire enclosing rectangle of the rho=2
Bernstein ellipse. Arc boxes cover the ellipse boundary and bound its
maximum; the maximum modulus principle handles the interior. The Lobatto
interpolant of degree n has uniform error at most 4*M*2^-n. Exact-node values,
discrete cosine coefficients and polynomial conversion are enclosed in Arb.
The effect on the normalized field L2 norm is at most A/(lambda*z) times
this uniform error. There is no boundary or ROI numerical quadrature in
the certified path.

Expand the polynomial in q=(distance/z)^2. Source reflection kills odd
powers, and the remaining coefficients use exact superellipse gamma moments.
This produces a global field polynomial in X=(x/5)^2 and Y=(y/5)^2.
The real and imaginary polynomials are each squared with a Kronecker stride
of 2n+1, so terms cannot alias; the two squares retain signed cross terms.
The normalized disk moment of X^i Y^j is
(1/2)_i (1/2)_j/(i+j+1)!. Summing the Gram polynomial encloses its exact
squared norm, including every rounding error. A negative lower endpoint is
replaced by zero using nonnegativity; a negative upper endpoint fails.
Add the validated uniform residual bound at norm level before squaring
to report the upper endpoint for the true squared norm.

The direct paired quadrature scales as n^6 before symmetry; the source
moment transform is O(n^4), and the disk Gram uses polynomial convolution
rather than six-dimensional quadrature. A certified low-rank decomposition
could reduce cost only with a uniform or operator-norm tail enclosure; it
is unverified here. Reported high-precision direct quadrature is empirical.
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--out",type=Path,default=Path("certified_bound_v6_results"))
    args=parser.parse_args()
    configs=[(("2","0.3","0.05"),12,192),(("12","10","0.3"),64,512),
             (("12","10","0.3"),80,640),(("12","10","0.3"),96,768)]
    print("Spectral/moment cases (condition, degree, bits):",configs,flush=True)
    if args.plan:
        return
    args.out.mkdir(parents=True,exist_ok=False)
    _,A,_,ctx=backend()
    proof=symbolic_proof()
    rows=[]
    for c,n,b in configs:
        try:
            row=certify(c,n,b)
        except (ArithmeticError,AssertionError,ValueError) as exc:
            row=dict(p=c[0],nf=c[1],theta=c[2],degree=n,bits=b,status="unverified",
                     reason=str(exc) or type(exc).__name__)
        rows.append(row)
        print(row,flush=True)
    ctx.prec=256
    _,geom=continuous(A)
    empirical=empirical_small()
    for sample in empirical:
        sample["within_small_certificate_displayed_interval"]=(
            rows[0].get("lower_display",1)<=float(sample["L2"])<=rows[0].get("upper_display",0))
    report=dict(proof=proof,records=rows,empirical_direct_small_case=empirical,
                geometric_upper=str(geom.upper()),empirical_baseline=4.678284775e-6,
                six_dimensional_direct_status="unverified; analytically factored, not executed",
                reproduction="python fresnel_certified_bound_v6.py --out NEW_DIRECTORY")
    (args.out/"proof.md").write_text(PROOF,encoding="utf-8")
    (args.out/"certificate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.out/"convergence.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(layout="constrained")
    baseline=[r for r in rows if r["p"]=="12" and "upper_display" in r]
    if baseline:
        ax.semilogy([r["degree"] for r in baseline],[r["upper_display"] for r in baseline],
                    "o-",label="Certified upper")
    ax.axhline(float(geom),linestyle="--",label="v3 geometric")
    ax.axhline(report["empirical_baseline"],linestyle=":",label="Empirical reference")
    ax.set(xlabel="Global interpolation degree",ylabel="Continuous raw residual L2 upper")
    ax.legend()
    fig.savefig(args.out/"convergence.png",dpi=170)


if __name__=="__main__":
    main()
