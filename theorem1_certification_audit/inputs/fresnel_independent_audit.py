"""Independent RS-I versus Padé audit. No imports from prior research code."""
import argparse
import csv
import json
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import roots_legendre, beta


BASE=("12","10","0.3","1")


def arb_kernel(q,k,C):
    i=C(0,1)
    b=16+12*q+q*q
    d=48+84*q+35*q*q
    if not all(x.real>0 for x in (1+q,b,d)):
        raise ArithmeticError("complex box cannot exclude branch or pole")
    t=(1+q).sqrt()
    # Independently derive both stable differences by symbolic subtraction.
    s=(t-1)**5
    phi=4*q*(q+2)/b
    # Use exact RS amplitude times the phase increment.
    ar=1/(1+q)+i/(k*t**3)
    gap=i*k*s/b
    return (i*k*phi).exp()*(i*s*(t*t+5*t+1)/(k*t**3*d)+ar*(gap.exp()-1))


def symbolic():
    import sympy as s
    t=s.symbols("t",positive=True)
    q=t*t-1
    assert s.cancel(t-1-4*q*(q+2)/(16+12*q+q*q)-(t-1)**5/(16+12*q+q*q))==0
    assert s.cancel(t**-3-(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48)
                    -(t-1)**5*(t*t+5*t+1)/(t**3*(48+84*q+35*q*q)))==0
    # Hand moments of a unit disk: E[r^(2k)]=1/(k+1).
    for d in (1,2,3):
        for k in range(d+1):
            assert s.simplify(s.gamma(k+s.Rational(1,2))*s.gamma(d-k+s.Rational(1,2))/
                (s.pi*s.factorial(d+1)) -
                s.Rational(comb(2*k,k)*comb(2*(d-k),d-k),
                           4**d*(d+1)*comb(d,k)))==0


def setup(A,condition):
    p,nf,theta,ar=map(A,condition)
    az=(25*A.pi()*(1+2/p).gamma()/(ar*(1+1/p).gamma()**2)).sqrt()
    ax,ay=ar*az,az
    z,lam=10/theta,10*theta/nf
    r=ax if p==2 else (ax**(2*p/(p-2))+ay**(2*p/(p-2)))**((p-2)/(2*p))
    q=((r+5)/z)**2
    Q=A(1)/1024
    while not Q>q:
        Q+=A(1)/1024
    return p,ax,ay,z,lam,2*A.pi()*z/lam,q,Q


def polynomial(A,C,P,K,Q,n):
    # Map the rho=2 ellipse to q. The full rectangle is checked for analyticity.
    h=Q/2
    arb_kernel(C(h+A(0,1)*5*h/4,A(0,1)*3*h/4),K,C)
    M=A(0)
    for i in range(512):
        s=(2*A.pi()*i/512).union(2*A.pi()*(i+1)/512)
        v=arb_kernel(C(h+5*h*s.cos()/4,3*h*s.sin()/4),K,C).abs_upper()
        M=M.union(v).upper()
    # Newton divided differences at analytic Lobatto nodes, not DCT/Clenshaw.
    xs=[(1+(A.pi()*i/n).cos())/2 for i in range(n+1)]
    ys=[arb_kernel(C(Q*x),K,C) for x in xs]
    for j in range(1,n+1):
        for i in range(n,j-1,-1):
            ys[i]=(ys[i]-ys[i-1])/(xs[i]-xs[i-j])
    poly=P([ys[-1]])
    for i in range(n-1,-1,-1):
        poly=poly*P([-xs[i],1])+P([ys[i]])
    return poly.coeffs(),(4*M/2**n).upper()


def source_shift(A,C,c,p,ax,ay,z,Q):
    """Average exact translations using derivative/moment operators."""
    n=len(c)-1
    # Coordinates in the input polynomial are normalized by z sqrt(Q).
    g=[((2*A(i)+1)/p).gamma() for i in range(n+1)]
    gd=[(1+(2*A(i)+2)/p).gamma() for i in range(n+1)]
    xp=[(ax*ax/(z*z*Q))**i for i in range(n+1)]
    yp=[(ay*ay/(z*z*Q))**i for i in range(n+1)]
    gp=[(25/(z*z*Q))**i for i in range(n+1)]
    terms={}
    bc=[[comb(2*k,2*i) for i in range(k+1)] for k in range(n+1)]
    for a in range(n+1):
        for b in range(n+1-a):
            moment=4*ax*ay*g[a]*g[b]*xp[a]*yp[b]/(p*p*gd[a+b])
            for t in range(a+b,n+1):
                scale=c[t]*moment
                for k in range(a,t-b+1):
                    key=k-a,t-k-b
                    v=scale*(comb(t,k)*bc[k][a]*bc[t-k][b])
                    terms[key]=terms.get(key,C(0))+v
    return {key:v*gp[sum(key)] for key,v in terms.items()}


def gram(A,R,terms):
    # Ordinary y-polynomials grouped by x degree; no Kronecker embedding.
    n=max(i+j for i,j in terms)
    re,im=[],[]
    for i in range(n+1):
        row=[terms.get((i,j)) for j in range(n+1-i)]
        re.append(R([v.real if v is not None else A(0) for v in row]))
        im.append(R([v.imag if v is not None else A(0) for v in row]))
    power=[A(1)]
    fac=[A(1)]
    for j in range(1,2*n+2):
        power.append(power[-1]*(A(j)-A(1)/2))
        fac.append(fac[-1]*j)
    total=A(0)
    for i in range(n+1):
        for k in range(i+1):
            poly=re[i]*re[k]+im[i]*im[k]
            total+=sum(v*power[i+k]*power[j]/fac[i+k+j+1]
                       for j,v in enumerate(poly.coeffs()))*(1 if i==k else 2)
    assert total.is_finite() and total.upper()>=0
    return total


def certify(condition,n,bits):
    from flint import arb as A, acb as C, acb_poly, arb_poly,ctx
    ctx.prec=bits
    start=perf_counter()
    p,ax,ay,z,lam,K,q,Q=setup(A,condition)
    if n==12:
        # Closed-form source/ROI examples for q^d on a unit circular aperture.
        # At the origin ∫ q^d = pi/(d+1), with z²*Q=1.
        for d in (1,2,3):
            terms=source_shift(A,C,[C(0)]*d+[C(1)],A(2),A(1),A(1),A(1),A(1))
            assert terms[0,0].real.overlaps(A.pi()/(d+1))
            # In complex source coordinates the circle leaves only z^k zbar^k.
            # The closed form is pi sum_k binom(d,k)^2 (x²+y²)^(d-k)/(k+1).
            for (i,j),value in terms.items():
                k=d-i-j
                exact=A.pi()*comb(d,k)**2*comb(d-k,i)*25**(i+j)/(k+1)
                assert value.real.overlaps(exact) and value.imag.contains(0)
        for d in (1,2,3):
            assert gram(A,arb_poly,{(d,0):C(1)}).overlaps(
                (A(2*d)+A(1)/2).gamma()/(A(1)/2).gamma()/A(2*d+2).gamma())
    coeff,error=polynomial(A,C,acb_poly,K,Q,n)
    squared=gram(A,arb_poly,source_shift(A,C,coeff,p,ax,ay,z,Q))/(lam*z)**2
    e=100*A.pi()*error/(lam*z)
    lo=squared.lower().sqrt()-e if squared.lower()>0 else A(0)
    hi=squared.upper().sqrt()+e
    return dict(p=condition[0],nf=condition[1],theta=condition[2],ar=condition[3],degree=n,bits=bits,
                q_max=str(q),q_interval_max=str(Q),lower=str(lo.lower() if lo>0 else A(0)),
                upper=str(hi.upper()),error=str(e.upper()),seconds=perf_counter()-start,
                status="interval-certified")


def direct(order,roi,method):
    start=perf_counter()
    p=12
    z,lam=100/3,.3
    K=2*np.pi*z/lam
    a=np.sqrt(100*np.pi/(4*beta(1/p,1+1/p)/p))
    def rule(n):
        x,w=roots_legendre(n)
        return (x+1)/2,w/2
    u,w=rule(order)
    # Angular/polar sectors, independent of v6 and the earlier ray-slope map.
    phi=np.pi*u/4
    edge=a*(np.cos(phi)**p+np.sin(phi)**p)**(-1/p)
    x=np.outer(u,edge*np.cos(phi)).ravel()
    y=np.outer(u,edge*np.sin(phi)).ravel()
    aw=np.tile(np.outer(u*w,w*np.pi/4*edge**2).ravel(),8)
    xs=np.concatenate((x,x,-x,-x,y,y,-y,-y))
    ys=np.concatenate((y,-y,y,-y,x,-x,x,-x))
    r,v=rule(roi)
    if method=="polar":
        X=np.outer(5*r,np.cos(np.pi*r/4)).ravel()
        Y=np.outer(5*r,np.sin(np.pi*r/4)).ravel()
        W=np.outer(2*r*v,v).ravel()
    else:
        X=np.repeat(5*(1-r*r),roi)
        Y=np.outer(5*r*np.sqrt(2-r*r),r).ravel()
        W=np.outer(8/np.pi*r*r*np.sqrt(2-r*r)*v,v).ravel()
    values=[]
    for x,y in zip(X,Y):
        q=((x-xs)**2+(y-ys)**2)/z**2
        t=np.sqrt(1+q)
        ph=K*(q/(1+t))**5/(16+12*q+q*q)
        amp=1/(1+q)+1j/(K*t**3)
        da=1j/K*(q/(t+1))**5*(t*t+5*t+1)/(t**3*(48+84*q+35*q*q))
        e=np.exp(1j*K*(q/2+q*q/4)/(1+3*q/4+q*q/16))
        vals=[np.sum(aw*g)/(1j*lam*z) for g in (
            e*(da+amp*np.expm1(1j*ph)),e*(da+amp*(1j*ph-ph**2/2)))]
        rho=np.sum(aw*abs(amp)*abs(ph)**3/6)/(lam*z)
        values.append((*vals,rho))
    N=np.sqrt(np.dot(W,abs(np.array(values))**2))
    return dict(method=method,aperture_order=order,roi_order=roi,precision="binary64",
                exact_norm=N[0],L2_taylor_norm=N[1],rho3_norm=N[2],oscillatory_sum=N[1]+N[2],
                seconds=perf_counter()-start,status="empirical")


def high_precision(order,roi):
    import mpmath as m
    start=perf_counter()
    with m.workdps(35):
        z,lam=m.mpf(100)/3,m.mpf(3)/10
        K=2*m.pi*z/lam
        a=m.sqrt(100*m.pi/(4*m.beta(m.mpf(1)/12,1+m.mpf(1)/12)/12))
        def rule(n):
            x,w=m.gauss_quadrature(n,"legendre")
            return [((u+1)/2,v/2) for u,v in zip(x,w)]
        ap,obs=rule(order),rule(roi)
        source=[]
        for t,w in ap:
            c,s=m.cos(m.pi*t/4),m.sin(m.pi*t/4)
            edge=a*(c**12+s**12)**(-m.mpf(1)/12)
            for r,v in ap:
                x,y=r*edge*c,r*edge*s
                aw=w*v*r*edge**2*m.pi/4
                source.extend((sx*xx,sy*yy,aw) for xx,yy in ((x,y),(y,x))
                              for sx in (-1,1) for sy in (-1,1))
        total=m.mpf(0)
        for r,v in obs:
            for t,w in obs:
                x,y=5*r*m.cos(m.pi*t/4),5*r*m.sin(m.pi*t/4)
                field=m.mpc(0)
                for u,s,aw in source:
                    q=((x-u)**2+(y-s)**2)/z**2
                    root=m.sqrt(1+q)
                    g=(q/(root+1))**5
                    amp=1/(1+q)+1j/(K*root**3)
                    da=1j*g*(root**2+5*root+1)/(K*root**3*(48+84*q+35*q*q))
                    field+=aw*m.exp(1j*K*(q/2+q*q/4)/(1+3*q/4+q*q/16))*(
                        da+amp*m.expm1(1j*K*g/(16+12*q+q*q)))
                total+=2*r*v*w*abs(field/(lam*z))**2
        return dict(method="mpmath polar",aperture_order=order,roi_order=roi,
                    precision="35 decimal digits",exact_norm=float(m.sqrt(total)),
                    norm_decimal=m.nstr(m.sqrt(total),34),status="empirical",
                    seconds=perf_counter()-start)


PROOF="""# Independent RS-I–Padé mathematical audit

For q=|r-s|²/z² and K=2πz/λ all fields share unit aperture illumination,
the prefactor 1/(iλz), and the omitted carrier exp(iK). RS-I has
amplitude (1+q)^-1+i(1+q)^-3/2/K and phase sqrt(1+q)-1. In the
physical derivative kernel z/R is the obliquity factor and 1/R is
spherical spreading. Fresnel uses unit amplitude and phase q/2.
Padé uses phase 4q(q+2)/(16+12q+q²) and amplitude
1/(1+q)+i(1+q/4-q²/48)/(K(1+7q/4+35q²/48)).
The exact RS-I minus Padé kernel is not the RS-I minus Fresnel kernel.
The target is sqrt(int_disk |U_RS-U_PP|²/(25π)), an absolute norm.

The continuous kernels are bounded on the compact product domains, so
Fubini expands the norm into int_ROI int_A int_A D(r,s) conj(D(r,t)).
For AR=ax/ay, the beta change of variables u=(|x|/ax)^p,v=(|y|/ay)^p
gives mu_2i,2j = 4 ax^(2i+1) ay^(2j+1) Gamma((2i+1)/p) Gamma((2j+1)/p) /
[p² Gamma(1+(2i+2j+2)/p)]. Set mu_00=100π. Independent reflection
eliminates odd source moments for all AR. After expanding q^d, apply
the translation/moment operator sum mu_ab ∂x^a ∂y^b/(a!b!).
The resulting polynomial has even independent powers and does not
require x/y exchange, including AR=2 or 4. Disk monomial moments of
X=(x/5)^2,Y=(y/5)^2 are (1/2)_i(1/2)_j/(i+j+1)! via beta integration.
Ordinary polynomial products at each X degree compute the real and
imaginary squares separately; signed cross terms are retained.
For a unit disk source and an axial observer, integrating q^d gives
π/(d+1) at z=1 for d=1,2,3. The corresponding disk checks for X^d
use elementary beta moments. Only at the end use Minkowski to add
the uniform kernel remainder times 100π/(λz). The exact squared norm
is nonnegative; a negative interval lower endpoint is safely replaced by zero.

For p>2 the enclosing source radius is the dual-norm expression
(ax^(2p/(p-2))+ay^(2p/(p-2)))^((p-2)/(2p)); p=2 uses max(ax,ay).
Adding ROI radius then dividing by z yields a safe q bound.
For Bernstein rho=2, Arb checks the enclosing rectangle is free of
the principal sqrt cut and poles. The denominator polynomials in q
also account for the Padé amplitude denominator. Full closed angular
intervals cover the ellipse boundary; the maximum modulus principle
extends the Arb maximum to the interior. The Chebyshev–Lobatto
interpolation error is <=4M rho^-n/(rho-1) (Trefethen, Approximation
Theory and Approximation Practice, Theorem 8.2; the analytic function
and magnitude bounds hold on the closed ellipse and in a neighborhood).
Source: https://people.maths.ox.ac.uk/trefethen/trefethen_sample.pdf
For complex values the same contour/aliasing proof uses the modulus.
This is interpolation, not truncation. Newton divided differences,
basis conversion, gamma and Gram products all use Arb enclosures.

Direct quadrature is empirical. The polar ROI uses octant symmetry
only at AR=1; the independent Cartesian comparison uses quadrant
reflection and the map x=5(1-u²), y=5u sqrt(2-u²) v. The 35-digit
calculation uses a separate implementation and its finite-order error
is not removed by precision. The original oscillatory value is
||L_2|| + ||rho_3||, a numerical estimate of a theoretical upper bound,
whereas the exact residual norm contains the complete exponential.

Reproduce: python fresnel_independent_audit.py --out NEW_DIRECTORY
"""


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("fresnel_independent_audit_results"))
    ap.add_argument("--plan",action="store_true")
    args=ap.parse_args()
    configs=[(BASE,n,max(256,8*n)) for n in (12,32,64,80,96)]
    configs += [((p,n,t,a),64,640) for p,n,t,a in (
        ("2","1","0.1","1"),("4","3","0.2","2"),("8","10","0.3","1"),
        ("16","1","0.1","4"),("12","3","0.2","4"),("16","10","0.3","4"))]
    print("Independent moment cases:",len(configs),flush=True)
    if args.plan:
        return
    args.out.mkdir(parents=True,exist_ok=False)
    symbolic()
    rows=[]
    for c,n,b in configs:
        try:
            row=certify(c,n,b)
        except (ArithmeticError,AssertionError,ValueError) as exc:
            row=dict(p=c[0],nf=c[1],theta=c[2],ar=c[3],degree=n,bits=b,status="unverified",
                     reason=str(exc) or type(exc).__name__,
                     reproduce=f"from fresnel_independent_audit import certify; print(certify({c!r},{n},{b}))")
        rows.append(row)
        print(c,n,row["status"],flush=True)
    ds=[]
    for a,r,method in ((24,24,"polar"),(32,32,"polar"),(40,32,"polar"),
                       (40,40,"polar"),(40,40,"Cartesian"),(40,48,"Cartesian"),
                       (40,56,"Cartesian")):
        ds.append(direct(a,r,method))
        print("Direct",a,r,method,ds[-1]["exact_norm"],flush=True)
    for a,r in ((8,8),(10,10)):
        ds.append(high_precision(a,r))
    for method in {r["method"] for r in ds}:
        series=[r for r in ds if r["method"]==method]
        for prev,row in zip(series,series[1:]):
            row["absolute_change"]=abs(row["exact_norm"]-prev["exact_norm"])
        for row in series:
            row["convergence_status"]=("within_1e-14_last_change" if row.get("absolute_change",1)<=1e-14
                                       else "not_established_by_this_row")
    # Existing result files are deliberately opened only after independent runs.
    old=json.loads(Path("certified_bound_v6_verified/certificate.json").read_text(encoding="utf-8"))
    old=next(r for r in old["records"] if r["p"]=="12" and r["degree"]==96)
    previous=json.loads(Path("fast_bound_quick_verified/fast_bound_report.json").read_text(
        encoding="utf-8"))["cases"][0]["B0_osc"][1]
    from flint import arb,ctx
    ctx.prec=768
    lo,hi=arb(old["lower"]).lower(),arb(old["upper"]).upper()
    for row in ds:
        value=arb(row["exact_norm"])
        row["inside_v6_interval"]=bool(lo<=value<=hi)
        row["distance_from_v6_interval"]=float(max(arb(0),lo-value,value-hi))
    baseline=rows[4]
    overlaps=bool(arb(baseline["lower"]).lower()<=hi and lo<=arb(baseline["upper"]).upper()) if "lower" in baseline else False
    last=ds[3]
    agrees=all(ds[i]["inside_v6_interval"] for i in (2,3,6))
    report=dict(definition_discrepancy="B" if agrees and abs(last["oscillatory_sum"]-previous)<1e-14 else "D",
                classification_detail="Same physical geometry, different target: additive Taylor bound versus exact residual norm.",
                independent_degree96_overlaps_v6=overlaps,old_oscillatory_sum=previous,
                gap_old_minus_direct_exact=previous-last["exact_norm"],
                taylor_norm_minus_direct_exact=last["L2_taylor_norm"]-last["exact_norm"],
                independently_computed_rho3_norm=last["rho3_norm"],
                independent_bound_sum_difference=last["oscillatory_sum"]-previous,
                certificates=rows,direct=ds,
                reproduction="python fresnel_independent_audit.py --out NEW_DIRECTORY")
    (args.out/"audit_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"proof.md").write_text(PROOF,encoding="utf-8")
    for name,data in (("certificates",rows),("direct_convergence",ds)):
        with (args.out/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            out=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            out.writeheader()
            out.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(layout="constrained")
    scope=[r for r in ds if "oscillatory_sum" in r]
    for key,label in (("exact_norm","Direct exact residual"),("oscillatory_sum","Direct Taylor norm + remainder")):
        ax.plot(range(len(scope)),[r[key]*1e6 for r in scope],"o-",label=label)
    ax.set_xticks(range(len(scope)),[f"{r['aperture_order']}/{r['roi_order']}\n{r['method']}" for r in scope])
    ax.set_ylabel("Normalized L2 in units of 1e-6")
    ax.legend()
    fig.savefig(args.out/"reference_discrepancy.png",dpi=170)


if __name__=="__main__":
    main()
