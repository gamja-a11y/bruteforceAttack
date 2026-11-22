"""Arb certificate for a piecewise Chebyshev boundary midpoint rule."""

import argparse
import json
from math import comb, factorial
from pathlib import Path
from time import perf_counter, process_time

from fresnel_error_bound_validation import symbolic_proof
from fresnel_quadrature_validation import write_csv


DEGREES = (8, 12, 16, 24, 32, 48, 64)
BOUNDARY = (16, 24, 32, 48, 64, 96)
BITS = (64, 128, 192, 256)


def backend():
    import flint
    from flint import acb, arb, ctx

    assert arb(1).gamma().contains(1)
    assert acb(-1).sqrt().imag.contains(1)
    assert acb(0).exp().real.contains(1)
    return flint, arb, acb, ctx


def verify(degree, panels, steps, roi, bits, m, condition):
    flint, A, C, ctx = backend()
    ctx.prec = bits
    started = perf_counter()
    p, nf, theta, ar = map(A, condition)
    pi = A.pi()
    area, z, lam = 100*pi, 10/theta, 10*theta/nf
    k = 2*pi*z/lam
    ay = (area*(1+2/p).gamma()/(4*ar*(1+1/p).gamma()**2)).sqrt()
    ax = ar*ay
    j = C(0,1)
    qmax = A(1)/2
    while not (((ax*ax+ay*ay).sqrt()+5)/z)**2 < qmax:
        qmax *= 2

    def kernel(q):
        if not (1+q.real > 0):
            raise ArithmeticError("Branch cut enclosure")
        t = (1+q).sqrt()
        d0, d1, d2 = 1+3*q/4+q*q/16, t**4+10*t*t+5, t**3*(35*t**4+14*t*t-1)
        if any(d.contains(0) for d in (d0,d1,d2)) or not (1+q).abs_lower() > 0:
            raise ArithmeticError("A denominator enclosure includes zero")
        phase = (q/2+q*q/4)/d0
        gap = (q/(t+1))**5
        delta = j*k*gap/d1
        return (j*k*phase).exp()*(j*gap*(t*t+5*t+1)/(k*d2)
                   +(1/(1+q)+j/(k*t**3))*sum(delta**i/factorial(i) for i in range(1,m+1)))

    def chebval(co, x):
        b = c = C(0)
        for coefficient in reversed(co[1:]):
            b,c = coefficient+2*x*b-c,b
        return co[0]+x*b-c

    pieces, cheb_rows = [], []
    offset, eps, sup = C(0), A(0), A(0)
    nodes = [(pi*i/degree).cos() for i in range(degree+1)]
    transform = [[(pi*i*l/degree).cos() for i in range(degree+1)] for l in range(degree+1)]
    for i in range(panels):
        a,b = qmax*i/panels,qmax*(i+1)/panels
        mid,h = (a+b)/2,(b-a)/2
        # Rectangle enclosing the entire Bernstein ellipse, rho=2.
        box = C(mid+A(0,1)*h*5/4,A(0,1)*h*3/4)
        maximum = kernel(box).abs_upper()
        error = 4*maximum/2**degree
        assert error.is_finite()
        eps = eps.union(error).upper()
        values = [kernel(C(mid+h*t)) for t in nodes]
        coeff = [sum(values[u]*row[u]/(2 if u in (0,degree) else 1)
                     for u in range(degree+1))*2/degree for row in transform]
        coeff[0] /= 2
        coeff[-1] /= 2
        sup = sup.union(sum(c.abs_upper() for c in coeff)).upper()
        # Integrate Chebyshev polynomials term by term in the local coordinate.
        anti = [C(0) for _ in range(degree+2)]
        anti[1] += coeff[0]*h
        anti[2] += coeff[1]*h/4
        for n in range(2,degree+1):
            anti[n+1] += coeff[n]*h/(2*(n+1))
            anti[n-1] -= coeff[n]*h/(2*(n-1))
        anti[0] -= chebval(anti,A(-1))
        pieces.append((a,b,offset,anti))
        offset += chebval(anti,A(1))
        cheb_rows.append(dict(segment=i,a=str(a),b=str(b),degree=degree,bits=bits,
                              uniform_error_upper=str(error.upper()),method="Arb Bernstein rectangle",certified=True))
    setup = perf_counter()-started
    support = A(0).union(qmax)

    def primitive(q):
        q = q.intersection(support)
        if not q.lower()>0:
            disk = A(0,1)*sup/2
            return C(disk,disk)
        mid = q.mid()
        # The exact primitive is continuous even at segment joins.
        values = []
        for a,b,constant,anti in pieces:
            segment = a.union(b)
            if segment.overlaps(mid):
                s = mid.intersection(segment)
                values.append(constant+chebval(anti,(2*s-a-b)/(b-a)))
        assert values
        re,im = values[0].real,values[0].imag
        for value in values[1:]:
            re,im = re.union(value.real),im.union(value.imag)
        # |H'| <= sup |P| / q_min; applies across piece boundaries by continuity.
        inflation = A(0,1)*sup*q.rad()/q.lower()
        return C(re/(2*mid)+inflation,im/(2*mid)+inflation)

    def sector(t):
        e = (1+t**p)**(-1/p)
        d = -e*t**(p-1)/(1+t**p)
        return e,t*e,d,e+t*d

    def integrand(g,x,y,swap,sx,sy):
        u,v,du,dv = g
        if swap:
            u,v,du,dv = v,u,dv,du
        dx,dy = sx*ax*u-x,sy*ay*v-y
        q = (dx**2+dy**2)/z**2
        # Reflections reverse orientation when their determinant is negative.
        det = sx*sy*(-1 if swap else 1)
        return primitive(q)*(dx*sy*ay*dv-dy*sx*ax*du)*det

    t0=perf_counter()
    geometry = []
    for i in range(steps):
        a,b=A(i)/steps,A(i+1)/steps
        geometry.append((sector(a.union(b)),sector((a+b)/2)))
    lower,upper,err2 = A(0),A(0),A(0)
    roi_rows = []
    for ir in range(roi):
        r0,r1=A(5*ir)/roi,A(5*(ir+1))/roi
        rad=r0.union(r1)
        for ia in range(roi):
            factor=4 if A(1)==ar else 2
            phi=(pi*ia/(factor*roi)).union(pi*(ia+1)/(factor*roi))
            x,y=rad*phi.cos(),rad*phi.sin()
            f,e = C(0),A(0)
            for g,mid in geometry:
                for swap in (False,True):
                    for sx in (-1,1):
                        for sy in (-1,1):
                            enclosure=integrand(g,x,y,swap,sx,sy)
                            sampled=integrand(mid,x,y,swap,sx,sy)
                            f+=sampled
                            e+=(enclosure-sampled).abs_upper()
            f /= steps*lam*z
            e /= steps*lam*z
            w=(r1*r1-r0*r0)/(25*roi)
            lower+=w*f.abs_lower()**2
            upper+=w*f.abs_upper()**2
            err2+=w*e**2
            roi_rows.append(dict(radial_cell=ir,angular_cell=ia,weight=str(w),
                                 midpoint_rule_field_abs_lower=str(f.abs_lower()),
                                 midpoint_rule_field_abs_upper=str(f.abs_upper()),
                                 boundary_error_upper=str(e.upper())))
    elapsed=perf_counter()-t0
    t0=perf_counter()
    # All retained even terms in J_n are positive. The disk norm uses
    # Minkowski with exact beta moments of individual monomials.
    n=5*(m+1)
    moment={}
    disk={}
    for i in range(n+1):
        for l in range(n+1-i):
            moment[i,l]=(4*ax**(2*i+1)*ay**(2*l+1)/p**2*((2*i+1)/p).gamma()
                         *((2*l+1)/p).gamma()/(1+(2*i+2*l+2)/p).gamma())
            a,b=A(2*i)+A(1)/2,A(2*l)+A(1)/2
            disk[i,l]=5**(2*(i+l))*(a.gamma()*b.gamma()/((a+b).gamma()*pi*(2*(i+l)+1))).sqrt()
    def jnorm(order):
        return sum(comb(order,k0)*comb(2*k0,2*i)*comb(2*(order-k0),2*l)
                   *moment[i,l]*disk[k0-i,order-k0-l]
                   for k0 in range(order+1) for i in range(k0+1)
                   for l in range(order-k0+1))/z**(2*order)

    rho=jnorm(n)*(1+1/k**2).sqrt()*(k/512)**(m+1)/(factorial(m+1)*lam*z)
    b0=(7/(1536*k)+k*(1+1/k**2).sqrt()/512)*jnorm(5)/(lam*z)
    ec=area*eps/(lam*z)
    eb=err2.sqrt()
    norm=upper.sqrt()
    total=(norm+ec+eb+rho).upper()
    assert all(v.is_finite() for v in (total,ec,eb,rho,lower,upper))
    values=dict(B_cert=total,chebyshev=ec.upper(),boundary=eb.upper(),
                roi_norm_upper=norm.upper(),roi_norm_lower=lower.sqrt().lower(),
                roi_norm_enclosure_width=(norm-lower.sqrt()).upper(),taylor=rho.upper(),
                geometric_B0_fallback=b0.upper())
    return ({key:str(value) for key,value in values.items()},
            {key:float(value) for key,value in values.items()},cheb_rows,roi_rows,
            dict(setup_seconds=setup,boundary_and_roi_seconds=elapsed,
                 taylor_seconds=perf_counter()-t0,total_seconds=perf_counter()-started,
                 flint=flint.__version__,q_max=str(qmax)))


def sampled_benchmark():
    from fresnel_error_bound_validation import Case, geometry, roi_rule
    from fresnel_fast_residual_bound import (
        area_field, boundary_field, boundary_geometry, make_primitive,
        moment_coefficients, rho_direct, rho_from_moments, timed,
    )
    import numpy as np

    case=Case(12.,10.,.3)
    rows=[]
    for count in (1,16,256,1024):
        if count==1:
            x,y=np.array([3.]),np.array([4.])
        else:
            x,y,_=roi_rule(int(count**.5),1.)

        def area():
            g=geometry(case.p,case.ar,32)
            return area_field(case,g,x,y,"L"),rho_direct(case,g,x,y)

        def cheb():
            g,_=boundary_geometry(case,32)
            primitive=make_primitive(case,32)
            return boundary_field(case,g,x,y,primitive),rho_from_moments(
                case,moment_coefficients(case),x,y)

        for label,fn in (("A_2D",area),("B_empirical_chebyshev",cheb)):
            _,wall,cpu=timed(fn,3)
            rows.append(dict(method=label,task="sampled_points",points=count,
                             total_seconds=wall,cpu_seconds=cpu,repetitions=3,
                             certification_status="empirically_validated_only"))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("plan","check-dependencies","quick","run","benchmark"):
        parser.add_argument("--"+key,action="store_true")
    parser.add_argument("--precision",type=int,choices=BITS,default=128)
    parser.add_argument("--roi-cells",type=int,choices=(2,4,8,16),default=2,
                        help="Number of cells on each normalized ROI axis")
    parser.add_argument("--out",type=Path,default=Path("certified_bound_quick"))
    args=parser.parse_args()
    print(f"Quick: 32 Chebyshev panels; degree 32; 16 midpoint panels/sector; "
          f"{args.roi_cells}x{args.roi_cells} ROI boxes.")
    try:
        print("Arb backend:",backend()[0].__version__)
    except ImportError:
        print("Install in an isolated environment: python -m pip install python-flint")
        return
    if args.plan or args.check_dependencies or not (args.quick or args.run or args.benchmark):
        return
    proof=symbolic_proof()
    args.out.mkdir(parents=True,exist_ok=False)
    primary=("12","10","0.3","1")
    configs=[(32,16,args.precision,2,primary),(32,16,args.precision,1,primary)]
    if args.run and not args.quick:
        configs += [(d,16,args.precision,2,primary) for d in DEGREES if d!=32]
        configs += [(32,n,args.precision,2,primary) for n in BOUNDARY if n!=16]
        configs += [(32,16,b,2,primary) for b in BITS if b!=args.precision]
        configs += [(32,16,args.precision,2,c) for c in (
            ("2","1","0.1","1"),("8","10","0.3","1"),("12","3","0.2","2"),("12","3","0.2","4"))]
    records,cheb,rois,budgets=[],[],[],[]
    for i,(d,n,b,m,c) in enumerate(configs):
        tag=dict(case=i,degree=d,boundary_midpoint_panels=n,bits=b,m=m,p=c[0],nf=c[1],theta=c[2],ar=c[3])
        try:
            cpu=process_time()
            balls,approx,samples,boxes,cost=verify(d,32,n,args.roi_cells,b,m,c)
            cost["cpu_seconds"]=process_time()-cpu
            records.append({**tag,**balls,**cost,"certification_status":"numerically_certified",
                            "display_approximations":approx})
            cheb.extend({**tag,**row} for row in samples)
            rois.extend({**tag,**row} for row in boxes)
            for key,value in balls.items():
                budgets.append({**tag,"component":key,"arb_enclosure":value,
                                "method":"Bernstein ellipse" if key=="chebyshev" else
                                "gamma moments and disk monomial norms" if key in ("taylor","geometric_B0_fallback") else
                                "outward-rounded boundary and ROI boxes"})
            budgets.extend(({**tag,"component":"arithmetic","arb_enclosure":None,
                             "method":"Included in all Arb enclosures; no separate assumed error"},
                            {**tag,"component":"ROI","arb_enclosure":balls["roi_norm_upper"],
                             "method":"Continuous ROI enclosed directly; lower endpoint recorded separately"}))
        except (ArithmeticError,AssertionError,ValueError) as exc:
            records.append({**tag,"certification_status":"incomplete","reason":str(exc) or type(exc).__name__})
        print(i,records[-1]["certification_status"],flush=True)
        if i==0 and records[-1]["certification_status"]!="numerically_certified":
            break
    benchmark=sampled_benchmark() if args.benchmark else []
    benchmark.extend({**{k:r[k] for k in ("case","total_seconds","cpu_seconds","certification_status")},
                      "method":"C_Arb","task":"continuous_ROI"}
                     for r in records if "total_seconds" in r)
    for name,rows in (("chebyshev_error",cheb),("boundary_quadrature",rois),("roi_norm",rois),
                      ("taylor_remainder",[r for r in budgets if r["component"]=="taylor"]),
                      ("error_budget",budgets),("bound_convergence",records),
                      ("bound_benchmark",benchmark)):
        write_csv(args.out/f"certified_{name}.csv",rows or [{"status":"incomplete"}])
    cert={"proof":proof,"records":records,"chebyshev_segments":cheb,"roi_cells":rois,
          "arithmetic":"Arb outward rounding; decimal display approximations are for plots only.",
          "scope":"Certified midpoint boundary and interval ROI enclosure, not the empirical GL evaluator.",
          "benchmark_scope":"Sampled A/B and continuous C are different tasks; no matched-accuracy speedup is claimed.",
          "benchmark":benchmark,
          "parameters":{"ellipse_rho":2,"panels":32,"roi_radial_cells":args.roi_cells,
                        "roi_angular_cells":args.roi_cells}}
    reference=Path(__file__).resolve().parent/"fast_bound_quick_verified/fast_bound_report.json"
    if reference.is_file():
        empirical=json.loads(reference.read_text(encoding="utf-8"))["cases"][0]["B0_osc"][1]
        cert["empirical_reference"]={"B0_osc2":empirical,"source":str(reference),
                                     "B_cert_to_empirical":records[0]["display_approximations"]["B_cert"]/empirical
                                     if "B_cert" in records[0] else None}
    for name in ("certificate","certified_bound_report"):
        (args.out/f"{name}.json").write_text(json.dumps(cert,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid=[r for r in records if "B_cert" in r]
    for key in ("bound_comparison","cost_breakdown","accuracy_speed_tradeoff"):
        fig,ax=plt.subplots()
        if valid:
            for field in (("B_cert","chebyshev","boundary","taylor") if key=="bound_comparison" else
                          ("setup_seconds","boundary_and_roi_seconds","taylor_seconds") if key=="cost_breakdown"
                          else ("B_cert",)):
                x=[r["total_seconds"] if key=="accuracy_speed_tradeoff" else r["case"] for r in valid]
                ax.semilogy(x,[r["display_approximations"].get(field,r.get(field)) for r in valid],"o-",label=field)
            ax.legend()
        ax.set_title(key.replace("_"," "))
        fig.savefig(args.out/f"certifi{'cation' if key=='cost_breakdown' else 'ed'}_{key}.png")
        plt.close(fig)


if __name__=="__main__":
    main()
