"""Continuous ROI certification from complex GL centers and real derivative bounds."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fresnel_certified_bound import backend
from fresnel_certified_bound_v2 import geometric
from fresnel_certified_bound_v3 import continuous, gauss, physics
from fresnel_error_bound_validation import symbolic_proof


def derivative_bound(A):
    """Bound |gradient L2| on the physical disk by differentiating G in q."""
    _,a,z,lam,k=physics(A)
    Q=A(5)/16
    radius=a*A(2).sqrt()+5
    assert ((radius/z)**2)<Q
    # s=sqrt(1+q)-1, D=16+12q+q², delta=s^5/D.
    s=Q/2
    delta=s**5/16
    dd=5*s**4/32+s**5*(12+2*Q)/16**2
    # A-Ap=i*s^5*N/(K*D2), N=t²+5t+1, D2=t³*(48+84q+35q²).
    t=(1+Q).sqrt()
    N=t*t+5*t+1
    D2prime=3*t*(48+84*Q+35*Q**2)/2+t**3*(84+70*Q)
    da=s**5*N/(48*k)
    dda=((5*s**4*N/2+s**5*A(7)/2)/48+s**5*N*D2prime/48**2)/k
    amp=(1+1/k**2).sqrt()
    amp_prime=1+3/(2*k)
    # phi_P'=phi'-delta', so its absolute value is <= 1/2+dd.
    alpha=k*delta
    polynomial=alpha+alpha**2/2
    gprime=(k*(A(1)/2+dd)*(da+amp*polynomial)+dda
            +amp_prime*polynomial+amp*k*dd*(1+alpha))
    return (100*A.pi()*gprime*2*radius/(lam*z**3)).upper(),Q


def make_center_solver(A,C,n,panels):
    _,a,z,lam,k=physics(A)
    nodes,weights=gauss(A,n)
    J=C(0,1)

    def geom(t):
        b=1+t**12
        if not b.real>0:
            raise ArithmeticError("sector branch enclosure")
        e=b**(-A(1)/12)
        d=-e*t**11/b
        return a*e,a*t*e,a*d,a*(e+t*d)

    def box(i):
        h=A(1)/(2*panels)
        return C((2*i+1)*h+A(0,1)*2*h,A(0,1)*2*h)

    geometry=[(geom(box(i)),[geom(C((i+t)/panels)) for t in nodes]) for i in range(panels)]
    inner=[(box(i),[C((i+t)/panels) for t in nodes]) for i in range(panels)]
    sectors=[(swap,sx,sy) for swap in (False,True) for sx in (-1,1) for sy in (-1,1)]

    @lru_cache(maxsize=None)
    def solve(radial_num,radial_den,angular_num,angular_den):
        r=A(5*radial_num)/radial_den
        phi=A.pi()*angular_num/(4*angular_den)
        x,y=r*phi.cos(),r*phi.sin()

        def f(g,u,swap,sx,sy):
            X,Y,dX,dY=g
            if swap:
                X,Y,dX,dY=Y,X,dY,dX
            dx,dy=sx*X-x,sy*Y-y
            q=u*(dx*dx+dy*dy)/z**2
            b,c,d=1+3*q/4+q*q/16,16+12*q+q*q,48+84*q+35*q*q
            if any(not v.real>0 for v in (1+q,b,c,d)):
                raise ArithmeticError("kernel branch/pole enclosure")
            t=(1+q).sqrt()
            g0=(q/(1+t))**5
            h=J*k*g0/c
            G=(J*k*(q/2+q*q/4)/b).exp()*(J*g0*(t*t+5*t+1)/(k*t**3*d)
               +(1/(1+q)+J/(k*t**3))*(h+h*h/2))
            return G*(dx*sy*dY-dy*sx*dX)*(sx*sy*(-1 if swap else 1))/2

        maximum,total=A(0),C(0)
        for gb,gs in geometry:
            for ub,us in inner:
                for swap,sx,sy in sectors:
                    maximum+=f(gb,ub,swap,sx,sy).abs_upper()
                    total+=sum(w*v*f(g,u,swap,sx,sy) for g,w in zip(gs,weights)
                               for u,v in zip(us,weights))
        error=(4*maximum/(2**(2*n-1)*panels**2*lam*z)).upper()
        total/=panels**2*J*lam*z
        assert total.is_finite() and error.is_finite()
        return total,error
    return solve


def certify_grid(A,solve,nr,na,L,rho):
    cells=[]
    norm2=lower2=mass=A(0)
    for i in range(nr):
        for j in range(na):
            z,e=solve(2*i+1,2*nr,2*j+1,2*na)
            r1=A(5*(i+1))/nr
            # Radial displacement plus arc displacement on the outer radius.
            distance=A(5)/(2*nr)+r1*A.pi()/(8*na)
            variation=(L*distance).upper()
            hi=z.abs_upper()+e+variation
            lo=z.abs_lower()-e-variation
            weight=A(2*i+1)/(nr*nr*na)
            mass+=weight
            norm2+=weight*hi**2
            if lo>0:
                lower2+=weight*lo**2
            slack=A(0,1)*(e+variation)
            cells.append(dict(radial=i,angular=j,weight=str(weight),gl_error=str(e),
                              spatial_error=str(variation),center_real=str(z.real),center_imag=str(z.imag),
                              cell_real=str(z.real+slack),cell_imag=str(z.imag+slack),
                              abs_upper=str(hi.upper()),status="interval-certified"))
    assert mass.contains(1)
    field=norm2.sqrt().upper()
    return cells,field,lower2.sqrt().lower(),(field+rho).upper()


PROOF="""# v4 audit and continuous ROI proof

## Audit of v3

Proven: the two Padé identities and nonnegative coefficient inequalities are
checked symbolically. The tighter geometric bound uses the chord above the
convex reciprocal product on its certified q range. Its nonnegative
pointwise majorant can be squared and integrated with exact disk moments.
The independent v2 gamma/disk computation overlaps the v3 original bound.

Proven: GL brackets are disjoint, lie strictly inside (-1,1), and exhibit
sign changes. There are exactly n roots because the polynomial has degree n.
The GL formula encloses positive weights. On each analytic parameter disk
of twice the real half-width, the degree-(2n-1) Taylor tail is bounded using
Cauchy's formula. The two-axis telescoping error is 4 M 2^-(2n-1) times
cell area, since the quadrature weights are positive and sum to one.
Arb branch/pole tests are required for all enclosing boxes.

The signed boundary normal is (dy,-dx), with the determinant of each
reflection/swap applied to its parametric orientation. Writing H as half
the unit-interval integral of G(uq) makes the q=0 formula regular.

## Continuous extension

Proven: for real q in the certified interval [0,Q], the derivative of G
is bounded by applying the product and quotient rules to the exact Padé
remainders. The code uses s<=Q/2, s'<=1/2 and positive lower bounds for
both denominators. The phase exponential has modulus one on the real axis.
Differentiation under the compact area integral gives
|grad L2|<=area*sup|G2'|*2*max|r-r'|/(lambda*z^3).

The physical field at a cell center is within the certified GL disk
about the Arb-enclosed complex sum. For every point of its polar cell,
the additional radius is the spatial Lipschitz bound times a radial-plus-arc
distance. The octant is valid only for this AR=1 case. Radial limits include
the circular edge exactly; there are no clipped Cartesian cells.
The normalized cell weights are (r1²-r0²)/(25*nangular) and sum to one.
Squaring the cell magnitude bounds, summing with these exact weights and
taking square roots encloses the continuous norm. Add the independent
Arb-certified Taylor remainder norm to bound the raw RS-I–PP field norm.

Arb arithmetic errors are already included in the enclosures. The final
comparison is against a rigorously enclosed geometric bound. The stored
oscillatory reference is empirical. Taking the smaller of two independently
certified upper endpoints is valid, but does not imply a phase-preserving
improvement. A finite schedule is not a guaranteed adaptive algorithm.
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--target",default="0.0005042324")
    parser.add_argument("--out",type=Path,default=Path("certified_bound_v4_results"))
    args=parser.parse_args()
    schedule=((1,1,12,24,128),(1,1,16,24,128),(2,1,16,24,128),(1,1,16,28,192))
    print("Finite schedule (radial cells, angular cells, GL order, panels, bits):",schedule,flush=True)
    if args.plan:
        return
    flint,A,C,ctx=backend()
    args.out.mkdir(parents=True,exist_ok=False)
    proof=symbolic_proof()
    records,cells=[],[]
    for nr,na,n,p,bits in schedule:
        ctx.prec=bits
        start=perf_counter()
        row=dict(radial_cells=nr,angular_cells=na,gl_order=n,panels=p,bits=bits)
        try:
            nodes,_=gauss(A,n)
            assert all(0<t<1 for t in nodes)
            original,geo=continuous(A)
            v2,rho=geometric(A)
            assert original.overlaps(v2)
            L,Q=derivative_bound(A)
            group,upper,lower,bound=certify_grid(A,make_center_solver(A,C,n,p),nr,na,L,rho.upper())
            cells.extend({**row,**r} for r in group)
            row.update(status="interval-certified",phase_bound=str(bound),display_bound=float(bound),
                       L2_upper=str(upper),L2_lower=str(lower),gradient_upper=str(L),q_upper=str(Q),
                       geometric_bound=str(geo.upper()),taylor_remainder=str(rho.upper()),
                       selected_upper=str((bound if bound<geo.upper() else geo.upper())),
                       improves_geometric=bool(bound<geo),target_met=bool(bound<A(args.target)))
        except (ArithmeticError,AssertionError,ValueError) as exc:
            row.update(status="unverified",reason=str(exc) or type(exc).__name__)
        row["seconds"]=perf_counter()-start
        records.append(row)
        print(row,flush=True)
        if row.get("target_met"):
            break
    report=dict(backend=flint.__version__,mathematical_proof=proof,records=records,cell_enclosures=cells,
                scope="AR=1, p=12, NF=10, theta=0.3, radius 5, m=2",
                empirical_reference=4.678284775e-6,requested_target=args.target,
                schedule_succeeded=any(r.get("target_met",False) for r in records),
                adaptive_status="unverified",speedup_status="unverified",
                reproduction="python fresnel_certified_bound_v4.py --out NEW_DIRECTORY")
    (args.out/"certificate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"proof.md").write_text(PROOF,encoding="utf-8")
    for name,rows in (("error_budget",records),("roi_cells",cells)):
        with (args.out/f"{name}.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(layout="constrained")
    done=[(i,r["display_bound"]) for i,r in enumerate(records) if "display_bound" in r]
    if done:
        ax.semilogy(*zip(*done),"o-",label="Continuous interval certificate")
    ax.axhline(5.0423241e-4,linestyle="--",label="v3 geometric upper (display)")
    ax.axhline(report["empirical_reference"],linestyle=":",label="Empirical reference")
    ax.set(xlabel="Configuration index",ylabel="Raw field L2 upper bound")
    ax.legend()
    fig.savefig(args.out/"convergence.png",dpi=170)


if __name__=="__main__":
    main()
