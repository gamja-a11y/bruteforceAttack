"""Certify conventional Fresnel and Padé residuals on the same continuous ROI."""
import argparse
import csv
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from fresnel_certified_bound_v6 import aperture_transform, disk_gram, physical


BASE=("12","10","0.3")


def residual(q,k,kind,C):
    i=C(0,1)
    if not (1+q).real>0:
        raise ArithmeticError("Square-root branch not excluded")
    t=(1+q).sqrt()
    amp=1/(1+q)+i/(k*t**3)
    phi=q/(1+t)
    if kind=="Fresnel":
        # phi - q/2 = -q²/(2(1+t)²); A-1 is also formed without subtraction.
        phase=q/2
        da=-q/(1+q)+i/(k*t**3)
        delta=-q*q/(2*(1+t)**2)
    else:
        phase_den=1+3*q/4+q*q/16
        amp_den=1+7*q/4+35*q*q/48
        if not (phase_den.real>0 and amp_den.real>0):
            raise ArithmeticError("Padé poles not excluded")
        phase=(q/2+q*q/4)/phase_den
        # Subtraction in high-precision complex balls encloses cancellation;
        # subtract only the imaginary amplitude terms (real terms coincide).
        da=i/k*(1/t**3-(1+q/4-q*q/48)/amp_den)
        delta=phi-phase
    return (i*k*phase).exp()*(da+amp*(i*k*delta).expm1())


def interpolate(A,C,P,k,Q,n,kind):
    h=Q/2
    residual(C(h+A(0,1)*5*h/4,A(0,1)*3*h/4),k,kind,C)
    M=A(0)
    for i in range(512):
        s=(2*A.pi()*i/512).union(2*A.pi()*(i+1)/512)
        q=C(h+5*h*s.cos()/4,3*h*s.sin()/4)
        M=M.union(residual(q,k,kind,C).abs_upper()).upper()
    nodes=[(1+(A.pi()*i/n).cos())/2 for i in range(n+1)]
    y=[residual(C(Q*x),k,kind,C) for x in nodes]
    for j in range(1,n+1):
        for i in range(n,j-1,-1):
            y[i]=(y[i]-y[i-1])/(nodes[i]-nodes[i-j])
    p=P([y[-1]])
    for i in range(n-1,-1,-1):
        p=p*P([-nodes[i],1])+P([y[i]])
    return p.coeffs(),(4*M/2**n).upper(),M


def peak_memory():
    if sys.platform=="win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("faults",wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in
                ("peak","working","peak_paged","paged","peak_nonpaged","nonpaged",
                 "pagefile","peak_pagefile")]
        handle=ctypes.windll.kernel32.GetCurrentProcess
        handle.restype=wintypes.HANDLE
        get=ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes=(wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD)
        c=Counters()
        c.cb=ctypes.sizeof(c)
        if get(handle(),ctypes.byref(c),c.cb):
            return int(c.peak)
        return None
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak*(1 if sys.platform=="darwin" else 1024))


def certify(condition,n,kind):
    from flint import arb as A, acb as C, acb_poly, arb_poly, ctx
    ctx.prec=max(256,8*n)
    start=perf_counter()
    p,nf,theta=map(A,condition)
    pi,a,z,lam,k,Q=physical(A,p,nf,theta)
    c,e,M=interpolate(A,C,acb_poly,k,Q,n,kind)
    terms=aperture_transform(A,C,c,p,a,z,Q)
    gram=disk_gram(A,arb_poly,terms)/(lam*z)**2
    slack=(100*pi*e/(lam*z)).upper()
    tentative=gram.lower().sqrt()-slack if gram.lower()>0 else A(0)
    lo=tentative.lower() if tentative>0 else A(0)
    hi=(gram.upper().sqrt()+slack).upper()
    assert hi.is_finite() and lo<=hi
    return dict(p=condition[0],nf=condition[1],theta=condition[2],ar=1,roi_radius=5,
                wavelength=str(lam),z=str(z),kind=kind,degree=n,bits=ctx.prec,
                q_upper=str(Q),ellipse_M_upper=str(M),
                kernel_error_upper=str(e),field_error_upper=str(slack),
                lower=str(lo),upper=str(hi),interval_width=str((hi-lo).upper()),
                lower_display=float(lo),upper_display=float(hi),width_display=float(hi-lo),
                seconds=perf_counter()-start,peak_process_bytes=peak_memory(),
                status="interval-certified",
                precision_flag="wide" if hi-lo>hi/100 else "within_1_percent_of_upper")


def direct_small():
    """Independent 40-digit direct quadrature for a circular NF=1 aperture."""
    import mpmath as mp

    out=[]
    with mp.workdps(40):
        z,lam=mp.mpf(100),mp.mpf(1)
        k=2*mp.pi*z/lam
        for n in (8,12,20,28):
            start=perf_counter()
            nodes,weights=mp.gauss_quadrature(n,"legendre")
            rule=[((x+1)/2,w/2) for x,w in zip(nodes,weights)]
            angular=[(mp.cos(mp.pi*u),w) for u,w in rule]
            norms=[mp.mpf(0),mp.mpf(0)]
            for r,wr in rule:
                fields=[mp.mpc(0),mp.mpc(0)]
                for s,ws in rule:
                    for c,w in angular:
                        q=(25*r*r+100*s*s-100*r*s*c)/z**2
                        t=mp.sqrt(1+q)
                        rs=(1/(1+q)+1j/(k*t**3))*mp.exp(1j*k*(t-1))
                        pp=(1/(1+q)+1j*(1+q/4-q*q/48)/(k*(1+7*q/4+35*q*q/48)))*mp.exp(
                            1j*k*(q/2+q*q/4)/(1+3*q/4+q*q/16))
                        fr=mp.exp(.5j*k*q)
                        fields[0]+=(rs-pp)*ws*w*s
                        fields[1]+=(rs-fr)*ws*w*s
                for j,v in enumerate(fields):
                    norms[j]+=2*r*wr*abs(v*200*mp.pi/(lam*z))**2
            out.append(dict(order=n,digits=40,status="empirical",seconds=perf_counter()-start,
                            Pade=mp.nstr(mp.sqrt(norms[0]),35),
                            Fresnel=mp.nstr(mp.sqrt(norms[1]),35)))
    return out


PROOF="""# Standard Fresnel extension

**Proven — physical definitions.** With unit illumination on the same
superellipse, let q=|r-s|²/z², k=2π/λ, K=kz and t=sqrt(1+q).
All three fields are exp(iK)/(iλz) times the aperture integral of their
carrier-stripped kernel. The common carrier is omitted for computation.
The exact scalar RS-I kernel is A exp(iK phi), with
A=1/(1+q)+i/(K t³), phi=q/(1+t). The projected spherical-spreading
factor in this convention is 1/(1+q). Padé retains the real amplitude
1/(1+q) and replaces the imaginary term by
i(1+q/4-q²/48)/(K(1+7q/4+35q²/48)), and replaces phi by
(q/2+q²/4)/(1+3q/4+q²/16).
Conventional Fresnel uses amplitude 1 and phase q/2.
Each error is an absolute raw complex field norm with uniform continuous
ROI measure dA/(25π), no phase fit or division by the RS-I norm.

**Proven — stable differences.** For either approximate amplitude B and
phase psi, D=exp(iK psi)[(A-B)+A expm1(iK(phi-psi))].
In particular, for Fresnel, A-1=-q/(1+q)+i/(K t³) and
phi-q/2=-q²/[2(1+t)²], by rationalizing t-1.
For Padé, the shared real amplitude cancels algebraically and the
remaining subtractions are formed in high precision balls. Arb encloses
any loss of precision. The v6 exact Padé formula is used only indirectly
through the archived result for a regression, not for calculating D.

**Interval-certified — approximation and norm.** The prior moment and
Gram implementations are reused, at AR=1. The dyadic domain enclosing
all physical q values comes from the largest source radius plus ROI radius.
On the whole rectangle enclosing the rho=2 ellipse, the code verifies
Re(1+q)>0 and, for Padé, excludes both rational poles by positive real
denominator enclosures. This verifies holomorphy through the ellipse.
Closed interval arcs cover its boundary; Arb encloses the maximum modulus
there and the maximum modulus principle covers the interior. The
degree-n Chebyshev–Lobatto interpolant error is 4 M rho^-n/(rho-1)
(Trefethen, ATAP, theorem 8.2), valid for complex values by the usual
aliasing argument with absolute values. This is the interpolation bound,
not the truncation bound.
All interpolant coefficients, beta/gamma source moments and Gram
operations are enclosed. Real/imaginary squares retain signed cross
terms, and their exact disk integral gives the continuous L2 squared.
With epsilon the uniform kernel error, triangle and reverse triangle
inequalities widen the polynomial norm by 100π epsilon/(λz), clipped
at zero on the lower side. These are two-sided enclosures.

**Empirical.** Direct 40-digit polar quadrature uses the three separate
kernels at finite orders in a circular case. Precision does not certify
spatial truncation. Timing and process high-water RSS are measurements,
not mathematical speedup statements. Novelty against the literature and
performance against a matched certifier remain unverified.

Theorem reference: https://people.maths.ox.ac.uk/trefethen/trefethen_sample.pdf
"""


def worker(c,n,kind):
    try:
        return certify(c,n,kind)
    except (ArithmeticError,AssertionError,ValueError) as exc:
        return dict(p=c[0],nf=c[1],theta=c[2],kind=kind,degree=n,status="unverified",
                    reason=str(exc) or type(exc).__name__)


def isolated(c,n,kind):
    cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker",*c,str(n),kind]
    start=perf_counter()
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    row=json.loads(result.stdout)
    row["worker_elapsed_seconds"]=perf_counter()-start
    return row


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--out",type=Path,default=Path("standard_fresnel_results"))
    parser.add_argument("--worker",nargs=5,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        p,nf,t,d,k=args.worker
        print(json.dumps(worker((p,nf,t),int(d),k),allow_nan=False))
        return
    configs=[(BASE,d) for d in (48,64,80,96)]
    configs += [(c,64) for c in (("2","1","0.1"),("4","3","0.2"),("8","10","0.3"),
                                  ("12","1","0.1"),("16","3","0.2"),("16","10","0.3"))]
    print(f"{len(configs)*2} isolated certificates, plus one empirical circular comparison.",flush=True)
    if args.plan:
        return
    from flint import arb,ctx
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for c,d in configs:
        for kind in ("Pade","Fresnel"):
            row=isolated(c,d,kind)
            rows.append(row)
            print(c,d,kind,row["status"],flush=True)
    for j in range(0,len(rows),2):
        a,b=rows[j:j+2]
        ctx.prec=768
        order="unverified"
        if "upper" in a and "upper" in b:
            if arb(a["upper"]).upper()<arb(b["lower"]).lower():
                order="Pade smaller"
            elif arb(b["upper"]).upper()<arb(a["lower"]).lower():
                order="Fresnel smaller"
            else:
                order="overlapping intervals"
        a["paired_comparison"]=b["paired_comparison"]=order
    direct=direct_small()
    for r in direct:
        for k in ("Pade","Fresnel"):
            c=next(x for x in rows if x["p"]=="2" and x["kind"]==k)
            r[f"{k}_inside_interval"]=(bool(arb(c["lower"]).lower()<=arb(r[k])<=arb(c["upper"]).upper())
                                       if "lower" in c else None)
    old=json.loads(Path("certified_bound_v6_verified/certificate.json").read_text(encoding="utf-8"))
    old=next(r for r in old["records"] if r["p"]=="12" and r["degree"]==96)
    bp=next(r for r in rows if r["p"]=="12" and r["degree"]==96 and r["kind"]=="Pade")
    regression=bool(arb(bp["lower"]).lower()<=arb(old["upper"]).upper()
                    and arb(old["lower"]).lower()<=arb(bp["upper"]).upper()) if "lower" in bp else False
    write_csv(args.out/"pade_vs_fresnel.csv",rows[:8])
    write_csv(args.out/"generalized_certification.csv",rows[8:])
    report=dict(records=rows,direct_empirical=direct,pade_v6_regression_intervals_overlap=regression,
                benchmark="Per-case isolated process peak RSS includes Python/imports; seconds exclude startup.",
                scope="Unit illumination, continuous radius-5 disk, AR=1; no phase alignment.")
    (args.out/"standard_fresnel_certificate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"standard_fresnel_proof.md").write_text(PROOF,encoding="utf-8")
    (args.out/"reproduction.md").write_text(
        "# Reproduction\n\nRun from the research directory with python-flint and mpmath installed.\n\n"
        "```bash\npython fresnel_standard_certification.py --plan\n"
        "python fresnel_standard_certification.py --out NEW_DIRECTORY\n```\n"
        "The run also uses the existing v6 source and archived result for the stated regression.\n"
        "Peak RSS is obtained per isolated worker and includes imports.\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,key,label in zip(axes,("upper_display","width_display"),("L2 upper endpoint","Enclosure width")):
        for kind in ("Pade","Fresnel"):
            series=[r for r in rows[:8] if r["kind"]==kind and key in r]
            ax.semilogy([r["degree"] for r in series],[r[key] for r in series],"o-",label=kind)
        ax.set(xlabel="Lobatto polynomial degree",ylabel=label)
        ax.legend()
    fig.savefig(args.out/"convergence.png",dpi=170)


if __name__=="__main__":
    main()
