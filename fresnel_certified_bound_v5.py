"""Arb-certified quadratic spatial models on a continuous disk ROI."""
import argparse
import csv
import json
import re
from math import comb, factorial
from pathlib import Path
from time import perf_counter

from fresnel_certified_bound import backend
from fresnel_certified_bound_v2 import geometric
from fresnel_certified_bound_v3 import continuous, gauss, physics
from fresnel_certified_bound_v4 import derivative_bound
from fresnel_error_bound_validation import symbolic_proof


POWERS=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))


def kernel_factory(A,C,k):
    from flint import acb_series
    I=C(0,1)

    def jets(q,order):
        b,c,d=1+3*q/4+q*q/16,16+12*q+q*q,48+84*q+35*q*q
        if any(not v.real>0 for v in (1+q,b,c,d)):
            raise ArithmeticError("unverified branch or denominator on quadrature box")
        u=acb_series([q,C(1)],order+1)
        t=(1+u).sqrt()
        g=(u/(1+t))**5
        h=I*k*g/(16+12*u+u*u)
        f=(I*k*(u/2+u*u/4)/(1+3*u/4+u*u/16)).exp()*(
            I*g*(t*t+5*t+1)/(k*t**3*(48+84*u+35*u*u))
            +(1/(1+u)+I/(k*t**3))*(h+h*h/2))
        return [f[i]*factorial(i) for i in range(order+1)]
    return jets


def real_derivatives(A,C,jets,Q):
    bounds=[A(0)]*4
    for i in range(256):
        q=(Q*i/256).union(Q*(i+1)/256)
        for j,value in enumerate(jets(C(q),3)):
            bounds[j]=bounds[j].union(value.abs_upper()).upper()
    return bounds


def prepare_solver(A,C,panels,order_cap):
    _,a,z,lam,k=physics(A)
    jets=kernel_factory(A,C,k)
    I=C(0,1)
    pref=1/(I*lam*z)

    def sector(t):
        b=1+t**12
        if not b.real>0:
            raise ArithmeticError("unverified superellipse branch")
        return b**(-A(1)/12)

    def box(i):
        h=A(1)/(2*panels)
        return C((2*i+1)*h+A(0,1)*2*h,A(0,1)*2*h)

    boxes=[box(i) for i in range(panels)]
    edges=list(map(sector,boxes))
    signs=[(swap,sx,sy) for swap in (False,True) for sx in (-1,1) for sy in (-1,1)]

    def solve(x,y):
        def integrate_node(t,e,r):
            # Positive-area octant map. Sum complex amplitudes before enclosure.
            X,Y=a*r*e,a*r*t*e
            result=[C(0) for _ in range(6)]
            for swap,sx,sy in signs:
                u,v=(Y,X) if swap else (X,Y)
                dx,dy=x-sx*u,y-sy*v
                q=(dx*dx+dy*dy)/z**2
                g,g1,g2=jets(q,2)
                qx,qy=2*dx/z**2,2*dy/z**2
                values=(g,g1*qx,g1*qy,g2*qx*qx+2*g1/z**2,g2*qx*qy,
                        g2*qy*qy+2*g1/z**2)
                result=[a0+b0 for a0,b0 in zip(result,values)]
            return [v*(a*a*r*e*e)*pref for v in result]

        maxima=[A(0)]*6
        for t,e in zip(boxes,edges):
            for r in boxes:
                maxima=[a0+b0.abs_upper() for a0,b0 in zip(maxima,integrate_node(t,e,r))]
        # Choose an order by a rigorous sufficient bound, capped by a budget.
        def remainder(n):
            return [(4*v/(2**(2*n-1)*panels**2)).upper() for v in maxima]
        n=next((n for n in range(6,order_cap+1)
                if all(v<A("1e-7") for v in remainder(n))),order_cap)
        nodes,weights=gauss(A,n)
        samples=[[C((i+t)/panels) for t in nodes] for i in range(panels)]
        edge_samples=[list(map(sector,ts)) for ts in samples]
        totals=[C(0)]*6
        for ts,es in zip(samples,edge_samples):
            for rs in samples:
                for t,e,w in zip(ts,es,weights):
                    for r,v in zip(rs,weights):
                        totals=[a0+w*v*b0 for a0,b0 in zip(totals,integrate_node(t,e,r))]
        errors=remainder(n)
        center=[v/panels**2 for v in totals]
        enclosed=[C(v.real+A(0,1)*e,v.imag+A(0,1)*e) for v,e in zip(center,errors)]
        assert all(v.is_finite() for v in enclosed+errors)
        return enclosed,errors,n
    return solve,jets


def local_model(A,C,cell,solve,third):
    r0,r1,t0,t1=cell
    pi=A.pi()
    phi0,phi1=t0*pi/4,t1*pi/4
    r,phi=(r0+r1)/2,(phi0+phi1)/2
    x,y=r*phi.cos(),r*phi.sin()
    jets,errors,n=solve(x,y)
    coefficients=[*jets[:3],jets[3]/2,jets[4],jets[5]/2]
    weight=(r1**2-r0**2)*(t1-t0)/25
    angle={}

    def trig_moment(i,j):
        key=i,j
        if key not in angle:
            result=C(0)
            I=C(0,1)
            for a in range(i+1):
                for b in range(j+1):
                    frequency=i+j-2*(a+b)
                    v=((I*frequency*phi1).exp()-(I*frequency*phi0).exp())/(I*frequency) if frequency else C(phi1-phi0)
                    result+=comb(i,a)*comb(j,b)*(-1)**b*v
            angle[key]=(result/(2**(i+j)*I**j)).real
        return angle[key]

    def moment(i,j):
        value=A(0)
        for a in range(i+1):
            for b in range(j+1):
                radial=(r1**(a+b+2)-r0**(a+b+2))/(a+b+2)
                value+=comb(i,a)*comb(j,b)*(-x)**(i-a)*(-y)**(j-b)*radial*trig_moment(a,b)
        return value*8/(25*pi)

    gram=A(0)
    for idx,(i,j) in enumerate(POWERS):
        for k,l in range_pairs(idx):
            a,b=POWERS[k]
            gram+=l*(coefficients[idx]*coefficients[k].conjugate()).real*moment(i+a,j+b)
    assert gram.upper()>=0
    poly=gram.upper().sqrt().upper()
    radius=(r1-r0)/2+r1*(phi1-phi0)/2
    remainder=(third*radius**3/6).upper()
    upper=(poly+weight.sqrt()*remainder).upper()
    assert upper.is_finite()
    return upper,dict(cell=[str(v) for v in cell],weight=str(weight),
                      derivatives=[str(v) for v in jets],gl_errors=[str(e) for e in errors],
                      order=n,polynomial_weighted_norm_upper=str(poly),
                      spatial_remainder_upper=str(remainder),local_weighted_upper=str(upper),
                      status="interval-certified")


def range_pairs(index):
    return [(k,1 if k==index else 2) for k in range(index+1)]


def gradient_replay(source,out):
    _,A,C,ctx=backend()
    ctx.prec=128
    saved=json.loads(source.read_text(encoding="utf-8"))
    # Reconstruct the bound independently rather than trust displayed decimals.
    _,rho=geometric(A)
    _,a,z,lam,k=physics(A)
    _,Q=derivative_bound(A)
    d=real_derivatives(A,C,kernel_factory(A,C,k),Q)
    L=(100*A.pi()*d[1]*2*(a*A(2).sqrt()+5)/(lam*z**3)).upper()
    rows=[]
    # The saved certificate uses only the root cell and its first bisection.
    for step in (0,1):
        group=[r for r in saved["cells"] if r["step"]==step and r["status"]=="interval-certified"]
        if len(group)!=step+1:
            continue
        if step==1:
            assert len(saved["history"])>=2 and saved["history"][1]["cells"]==2
        total=mass=A(0)
        for row in group:
            r0,r1,t0,t1=map(A,row["cell"])
            real,imag=re.fullmatch(r"(\[.*?\]) [+-] (\[.*?\])j",
                                   row["derivatives"][0]).groups()
            # Only the magnitude is used, so the printed sign of im is immaterial.
            field=C(A(real),A(imag))
            weight=(r1**2-r0**2)*(t1-t0)/25
            distance=(r1-r0)/2+r1*A.pi()*(t1-t0)/8
            total+=weight*(field.abs_upper()+L*distance)**2
            mass+=weight
        assert mass.contains(1)
        rows.append(dict(cells=len(group),upper_ball=str((total.sqrt()+rho.upper()).upper()),
                         display=float((total.sqrt()+rho.upper()).upper()),
                         status="interval-certified",method="first-order fallback"))
    out.mkdir(parents=True,exist_ok=False)
    (out/"gradient_fallback.json").write_text(json.dumps(dict(
        source=str(source),gradient_upper=str(L),records=rows,
        proof="Straight segments remain in the convex disk. Add L times radial-plus-arc distance "
              "to each certified center radius, integrate weighted squares, add the phase remainder."),
        indent=2),encoding="utf-8")
    print(rows)


PROOF="""# v5 proof and scope

**Proven — v4 audit.** Put s=sqrt(1+q)-1. On real q in [0,Q], s<=Q/2,
s'<=1/2 and the positive Padé denominators are bounded below by their
values at zero. Applying product and quotient rules gives the v4 G'
majorant, with |exp(iK phi_P)|=1. Converting q derivatives to spatial
derivatives contributes 2*distance/z² and the field prefactor contributes
1/(lambda*z). The disk measure is dA/(25*pi), or 8*dA/(25*pi) on its octant.

**Proven — spatial differentiation.** q_x=2(x-x')/z²,
q_y=2(y-y')/z², q_xx=q_yy=2/z², q_xy=0.
U_i integrates G' q_i; U_ij integrates G'' q_i q_j+G' q_ij.
The aperture is fixed. All differentiated kernels are continuous on the
compact product of the closed aperture and a neighborhood of the ROI;
positive real denominators and the principal square-root branch provide
a uniform majorant. Dominated differentiation therefore applies.
Arb power series compute G derivatives of the exact stabilized Padé expression.

**Interval-certified — centers.** The area map uses eight signed/swapped
copies of x'=a*r*e(t), y'=a*r*t*e(t), with Jacobian a²*r*e(t)².
The same positive GL root/weight certificate as v3 is used. On each
complex parameter polydisk all branch and pole conditions are checked.
Cauchy and product-rule telescoping give 4*M*2^-(2n-1) times cell area
for each of the six integrands. Complex arithmetic is accumulated before
adding the error disks. Area GL is used here instead of boundary GL.

**Proven — local remainder and norm.** Along any unit segment in the disk,
|q'|<=2*distance/z², q''=2/z², q'''=0. Thus the third directional
derivative is bounded by area/(lambda*z) times
sup|G'''|*(2*distance/z²)^3+3*sup|G''|*(2*distance/z²)*(2/z²).
Real q interval subdivision certifies these suprema. The Taylor remainder
is <=M3*distance³/6; distance is bounded by radial plus outer-radius arc
distance. Polar cells cover the circular edge exactly. The quadratic
Taylor model is squared as a complex polynomial and integrated via exact
polar moments, including its signed cross terms. Add the local remainder
with Minkowski, combine disjoint weighted cell squares, then add the
certified Padé phase remainder norm. Arb encloses arithmetic throughout.

Largest-upper-contribution bisection is budgeted, not a guarantee of hitting
the requested tolerance. The geometric and continuous phase certificates
bound the same raw field norm. The oscillatory reference remains empirical.
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--splits",type=int,default=1)
    parser.add_argument("--panels",type=int,default=24)
    parser.add_argument("--order-cap",type=int,default=16)
    parser.add_argument("--target",default="0.0005042324")
    parser.add_argument("--out",type=Path,default=Path("certified_bound_v5_results"))
    parser.add_argument("--gradient-replay",type=Path,help="First-order comparison for a saved 0/1-split certificate")
    args=parser.parse_args()
    if args.splits<0 or args.panels<1 or args.order_cap<6:
        parser.error("Require nonnegative splits, positive panels and order cap >=6")
    print(f"At most {1+2*args.splits} certified centers; {args.panels} panels/axis; GL cap {args.order_cap}",flush=True)
    if args.plan:
        return
    if args.gradient_replay:
        gradient_replay(args.gradient_replay,args.out)
        return
    flint,A,C,ctx=backend()
    ctx.prec=128
    ctx.cap=8
    args.out.mkdir(parents=True,exist_ok=False)
    proof=symbolic_proof()
    _,geo=continuous(A)
    _,rho=geometric(A)
    _,a,z,lam,_=physics(A)
    L,Q=derivative_bound(A)
    solve,jets=prepare_solver(A,C,args.panels,args.order_cap)
    d=real_derivatives(A,C,jets,Q)
    scale=2*(a*A(2).sqrt()+5)/z**2
    # Independent direct series derivative check of the same global v4 inequality.
    new_L=(100*A.pi()*d[1]*scale/(lam*z)).upper()
    third=(100*A.pi()*(d[3]*scale**3+3*d[2]*scale*2/z**2)/(lam*z)).upper()
    t0=perf_counter()
    cells,history,log=[],[],[]
    pending=[(A(0),A(5),A(0),A(1))]
    for step in range(args.splits+1):
        failed=False
        for cell in pending:
            start=perf_counter()
            try:
                upper,row=local_model(A,C,cell,solve,third)
                cells.append((cell,upper,row))
                log.append({**row,"step":step,"seconds":perf_counter()-start})
            except (ArithmeticError,AssertionError,ValueError) as exc:
                log.append(dict(cell=[str(v) for v in cell],step=step,seconds=perf_counter()-start,
                                status="unverified",reason=str(exc) or type(exc).__name__))
                failed=True
                break
        if failed:
            break
        mass=sum((c[1]**2-c[0]**2)*(c[3]-c[2])/25 for c,_,_ in cells)
        assert mass.contains(1)
        bound=(sum(upper**2 for _,upper,_ in cells).sqrt()+rho.upper()).upper()
        row=dict(step=step,cells=len(cells),upper_ball=str(bound),display=float(bound),
                 meets_target=bool(bound<A(args.target)),improves_geometric=bool(bound<geo),
                 elapsed_seconds=perf_counter()-t0,status="interval-certified")
        if history:
            row["display_decrease"]=history[-1]["display"]-row["display"]
            row["incremental_seconds"]=row["elapsed_seconds"]-history[-1]["elapsed_seconds"]
        history.append(row)
        print(row,flush=True)
        if row["meets_target"] or step==args.splits:
            break
        # Split the largest weighted contribution along its longer physical dimension.
        index=max(range(len(cells)),key=lambda i:float(cells[i][1]))
        (r0,r1,t0a,t1a),_,_=cells.pop(index)
        if r1-r0>=r1*A.pi()*(t1a-t0a)/4:
            mid=(r0+r1)/2
            pending=[(r0,mid,t0a,t1a),(mid,r1,t0a,t1a)]
        else:
            mid=(t0a+t1a)/2
            pending=[(r0,r1,t0a,mid),(r0,r1,mid,t1a)]
    report=dict(backend=flint.__version__,proof=proof,geometric_upper=str(geo.upper()),
                taylor_phase_remainder=str(rho.upper()),v4_global_gradient=str(L),
                independent_series_gradient_upper=str(new_L),third_directional_upper=str(third),
                requested_target=args.target,empirical_reference=4.678284775e-6,
                history=history,cells=log,adaptive_guarantee="unverified",
                scope="p=12, NF=10, theta=0.3, AR=1, radius=5, Taylor residual m=2",
                reproduce="python fresnel_certified_bound_v5.py --out NEW_DIRECTORY")
    (args.out/"certificate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"proof.md").write_text(PROOF,encoding="utf-8")
    for name,rows in (("convergence",history),("cell_error_budget",log)):
        with (args.out/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(layout="constrained")
    if history:
        ax.semilogy([r["cells"] for r in history],[r["display"] for r in history],"o-",label="Quadratic model")
    ax.axhline(float(geo),linestyle="--",label="Geometric certificate")
    ax.set(xlabel="Active ROI cells",ylabel="Continuous residual norm upper")
    ax.legend()
    fig.savefig(args.out/"convergence.png",dpi=160)


if __name__=="__main__":
    main()
