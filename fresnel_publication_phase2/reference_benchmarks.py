"""Reference calculations for the pilot's circular Fresnel residual, G1."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from adaptive_certification import peak_mib, decimal_out, rational


def empirical(order,degree):
    import numpy as np
    from scipy.special import roots_legendre
    x,w=roots_legendre(order)
    x,w=(x+1)/2,w/2
    weight=2*x*w
    ru=x[:,None,None]*5
    sv=x[None,:,None]*10
    q=(ru**2+sv**2-2*ru*sv*np.cos(np.pi*x)[None,None,:])/100**2
    K=200*np.pi
    if degree:
        from flint import acb_series as S, acb as C, arb as A, ctx
        with ctx.workprec(256):
            old=ctx.cap
            try:
                ctx.cap=degree+1
                v=S([0,1])
                k=200*A.pi()
                t=(1+v).sqrt()
                s=(1/(1+v)+C(0,1)/(k*t**3))*((t-1)*C(0,k)).exp()-(v*C(0,k)/2).exp()
                coeff=[complex(s[i]) for i in range(degree+1)]
            finally:
                ctx.cap=old
        values=np.polynomial.polynomial.polyval(q,coeff)
    else:
        t=np.sqrt(1+q)
        rs=(1/(1+q)+1j/(K*t**3))*np.exp(1j*K*q/(1+t))
        values=rs-np.exp(0.5j*K*q)
    field=np.pi*np.sum(values*weight[None,:,None]*w[None,None,:],axis=(1,2))
    return dict(estimate=float(np.sqrt(weight@abs(field)**2)),status="empirical",degree=degree or None,
                quadrature_order=order,evaluations=order**3)


def range_quadrature(n):
    # Uniform cells in squared normalized radii and half azimuth.
    # Their source and ROI area weights are exactly 1/n per variable.
    from flint import arb as A, acb as C, ctx
    with ctx.workprec(192):
        pi=A.pi()
        k=200*pi
        i=C(0,1)
        uv=[(A(j)/n).union(A(j+1)/n) for j in range(n)]
        root=[(A(j)/n).sqrt().union((A(j+1)/n).sqrt()) for j in range(n)]
        angles=[(pi*t).cos() for t in uv]
        low,high=A(0),A(0)
        for a in root:
            field=C(0)
            for b in root:
                for c in angles:
                    q=(25*a*a+100*b*b-100*a*b*c)/10000
                    q=q.max(A(0)).min(A(225)/10000)
                    t=(1+q).sqrt()
                    amp=1/(1+q)+i/(k*t**3)
                    gap=-q*q/(2*(1+t)**2)
                    diff=-q/(1+q)+i/(k*t**3)
                    field+=(i*k*q/2).exp()*(diff+amp*(i*k*gap).expm1())
            field*=pi/(n*n)
            low+=field.abs_lower()**2/n
            high+=field.abs_upper()**2/n
        lo=(low.max(A(0))).sqrt().lower()
        hi=high.sqrt().upper()
        if not (lo.is_finite() and hi.is_finite() and hi>=lo>=0):
            raise ValueError("Riemann enclosure failed")
        def frac(x):
            m,e=x.man_exp()
            return rational([str(m),int(e)])
        return dict(lower=decimal_out(frac(lo),False),upper=decimal_out(frac(hi),True),
                    absolute_width=decimal_out(frac(hi)-frac(lo),True),quadrature_order=n,
                    status="interval range quadrature",evaluations=n**3)


def main():
    if sys.argv[1:2]==["--worker"]:
        method,n,d=sys.argv[2:]
        start=time.perf_counter()
        row=range_quadrature(int(n)) if method=="range" else empirical(int(n),int(d))
        row.update(kernel_s=time.perf_counter()-start,peak_worker_mib=peak_mib())
        print(json.dumps(row))
        return
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    for rep in range(1,4):
        jobs=[("direct",n,0) for n in (12,20,32)]+[("Taylor",32,n) for n in (16,24,32)]+[("range",n,0) for n in (4,8,16)]
        for method,n,d in jobs[rep:]+jobs[:rep]:
            start=time.perf_counter()
            p=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",method,str(n),str(d)],
                             capture_output=True,text=True,check=True)
            rows.append(dict(method=method,rep=rep,seconds=time.perf_counter()-start,**json.loads(p.stdout)))
    with args.out.open("x",encoding="utf8") as f:
        json.dump(rows,f,indent=2)


if __name__=="__main__":
    main()

