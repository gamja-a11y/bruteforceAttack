"""Symbolic and numerical audit of the Padé–Padé field and sensitivity bounds."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, xlogy

from fresnel_quadrature_validation import unit_legendre, weighted_norm, write_csv
from fresnel_roi_convergence import polar_rule, verify_roi
from fresnel_shape_derivative_validation import stencil


RADIUS, ROI = 10., 5.
AREA = np.pi*RADIUS**2
LEVELS = ((32,32,48),(48,32,64),(64,32,96),(64,40,96))
HS = (.02,.01,.005,.0025)
REL_TOL = 1e-3


@dataclass(frozen=True)
class Case:
    p: float
    nf: float
    theta: float
    ar: float = 1.

    @property
    def z(self):
        return RADIUS/self.theta

    @property
    def wavelength(self):
        return RADIUS*self.theta/self.nf

    @property
    def k(self):
        return 2*np.pi*self.z/self.wavelength

    @property
    def pref(self):
        return 1/(1j*self.wavelength*self.z)

    @property
    def constant(self):
        return 7/(1536*self.k)+self.k*np.hypot(1,1/self.k)/512

    def metadata(self):
        return {**vars(self),"wavelength":self.wavelength,"z":self.z,"K":self.k,"C_K":float(self.constant)}


def symbolic_proof():
    import sympy as s

    t, u = s.symbols("t u", positive=True)
    q = t*t-1
    dphi = t**4+10*t*t+5
    damp = t**3*(35*t**4+14*t*t-1)
    for residual in (
        (t-1)-(q/2+q*q/4)/(1+3*q/4+q*q/16)-(t-1)**5/dphi,
        t**-3-(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48)
        -(t-1)**5*(t*t+5*t+1)/damp,
    ):
        if s.cancel(residual) != 0:
            raise ArithmeticError("Supplied Padé identity failed")
    polys = {
        "positive_amplitude_denominator": damp,
        "phase_inequality": (t+1)**5*dphi-512,
        "amplitude_inequality": 7*(t+1)**5*damp-1536*(t*t+5*t+1),
    }
    checks = {}
    for name, polynomial in polys.items():
        coefficients = s.Poly(polynomial.subs(t,u+1),u).all_coeffs()
        if any(c < 0 for c in coefficients) or (
            name=="positive_amplitude_denominator" and coefficients[-1]<=0
        ):
            raise ArithmeticError(f"Nonnegativity certificate failed: {name}")
        checks[name] = [int(c) for c in coefficients]
    return {
        "symbolic_identities_verified": True,
        "range": "All real q >= 0 and K > 0, with t=sqrt(1+q)>=1; no upper restriction on q.",
        "polynomial_coefficients_descending_in_t_minus_1": checks,
        "kernel": "Write Delta F=e^(iK phi_PP)[A-A_PP+A(expm1(iK Delta phi))]. "
                  "Use |expm1(iu)|<=|u| for real u and |A|<=sqrt(1+1/K^2).",
        "field_bounds": "Triangle inequality pointwise in the area integral for B0 and in "
                        "the fixed-AR boundary transport integral for B1; then use monotonicity of the ROI L2 norm.",
        "prediction_bounds": "Reverse triangle inequality gives |D-D_tilde|<=B0. "
                             "Also |N-||U_PP||||<=B0, so if B0<||U_PP|| then "
                             "N>=||U_PP||-B0>0, yielding the practical common-factor bound.",
        "sensitivity_bound": "In the real inner product Re<.,.>, derivative of ||X|| is "
                             "Re<X/||X||,X'> at nonzero X. For X=U_RS-U_F and Y=U_PP-U_F, "
                             "||X/||X||-Y/||Y||||<=2||X-Y||/min(||X||,||Y||) "
                             "by the triangle and reverse triangle inequalities. Hence "
                             "|D'-D_tilde'|<=B1+2B0||Y'||/min(D,D_tilde). "
                             "The quotient rule and |N'|<=||U_RS'|| give the proposed BS. "
                             "Requires differentiable fields and N,D,D_tilde>0.",
    }


def kernels(q,k,delta_only=False):
    t = np.sqrt(1+q)
    phi = q/(t+1)
    phi_p = (q/2+q*q/4)/(1+3*q/4+q*q/16)
    a = t**-2+1j/(k*t**3)
    gap = phi**5
    phase_error = gap/(t**4+10*t*t+5)
    amplitude_error = 1j/k*gap*(t*t+5*t+1)/(t**3*(35*t**4+14*t*t-1))
    phase_p = np.exp(1j*k*phi_p)
    delta = phase_p*(amplitude_error+a*np.expm1(1j*k*phase_error))
    if delta_only:
        return delta
    ap = t**-2+1j/k*(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48)
    return a*np.exp(1j*k*phi), np.exp(.5j*k*q), ap*phase_p, delta


def half_axes(p,ar):
    factor = 4*np.exp(2*gammaln(1+1/p)-gammaln(1+2/p))
    ay = np.sqrt(AREA/(ar*factor))
    return ar*ay, ay


def mu(m,n,p,ax,ay):
    return np.exp(np.log(4/p**2)+(2*m+1)*np.log(ax)+(2*n+1)*np.log(ay)
                  +gammaln((2*m+1)/p)+gammaln((2*n+1)/p)-gammaln(1+(2*m+2*n+2)/p))


def j5_moments(case,x,y):
    ax,ay = half_axes(case.p,case.ar)
    j = np.zeros_like(x)
    # Parity kills odd source moments. All surviving terms are nonnegative,
    # including at the axis, so this form has no alternating-sum cancellation.
    for k in range(6):
        for m in range(k+1):
            for n in range(6-k):
                j += (comb(5,k)*comb(2*k,2*m)*comb(10-2*k,2*n)
                      *x**(2*k-2*m)*y**(10-2*k-2*n)*mu(m,n,case.p,ax,ay))
    return j/case.z**10


def geometry(p,ar,order,boundary=False):
    ax,ay = half_axes(p,ar)
    t,wt = unit_legendre(order)
    if boundary:
        base = (digamma(1+1/p)-digamma(1+2/p))/p**2

        def log_derivative(s):
            u = s**p
            return base+np.log1p(u)/p**2-xlogy(u,s)/(p*(1+u))

        # Split where the normal velocity changes sign, so |velocity| is smooth
        # on each quadrature interval. In scaled coordinates that root is shared.
        root = brentq(log_derivative,0,1)
        t,wt = np.r_[root*t,root+(1-root)*t], np.r_[root*wt,(1-root)*wt]
    edge = (1+t**p)**(-1/p)
    if boundary:
        x,y = edge,t*edge
        w = ax*ay*edge**2*wt*log_derivative(t)
    else:
        r,wr = unit_legendre(order)
        x,y = np.outer(r,edge).ravel(),np.outer(r,t*edge).ravel()
        w = ax*ay*np.outer(r*wr,wt*edge**2).ravel()
    xs = ax*np.concatenate((x,-x,-x,x,y,-y,-y,y))
    ys = ay*np.concatenate((y,y,-y,-y,x,x,-x,-x))
    w = np.tile(w,8)
    invariant = abs(w.sum()) if boundary else abs(w.sum()-AREA)
    if invariant > AREA*2e-6:
        raise RuntimeError(f"Area conservation check failed at p={p}, AR={ar}, order={order}")
    return xs,ys,w


@lru_cache(maxsize=None)
def roi_rule(order,ar):
    if ar==1:
        rule = polar_rule(order,order)
    else:
        t,w = unit_legendre(order)
        angle = np.pi/2*t
        rule = (ROI*np.outer(t,np.cos(angle)).ravel(),ROI*np.outer(t,np.sin(angle)).ravel(),
                np.outer(2*t*w,w).ravel())
    verify_roi(*rule)
    return rule


def integrate(case,order,roi,boundary=False,delta_only=False):
    x,y,_ = roi_rule(roi,case.ar)
    xa,ya,weights = geometry(case.p,case.ar,order,boundary)
    values = np.empty((1 if delta_only else 6,x.size),complex)
    for i in range(0,x.size,6):
        s = slice(i,i+6)
        q = ((x[s,None]-xa)**2+(y[s,None]-ya)**2)/case.z**2
        if delta_only:
            values[0,s] = np.sum(kernels(q,case.k,delta_only=True)*weights,axis=1)*case.pref
            continue
        rs,uf,pp,d = kernels(q,case.k)
        values[:4,s] = np.array([np.sum(value*weights,axis=1) for value in (rs,uf,pp,d)])*case.pref
        values[4,s] = np.sum(q**5*abs(weights),axis=1)
        values[5,s] = np.sum(abs(d)*abs(weights),axis=1)*abs(case.pref)
    return values


@lru_cache(maxsize=128)
def fd_field(case,order,roi):
    return integrate(case,order,roi,delta_only=True)[0]


def norm_derivative(field,derivative,w):
    return float(np.vdot(field,w*derivative).real/weighted_norm(field,w))


def ratio(bound,actual):
    return float(bound/actual) if actual>1e-14 else None


def evaluate(case,levels):
    order,roi,border = levels
    x,y,w = roi_rule(roi,case.ar)
    bulk = integrate(case,order,roi)
    boundary = integrate(case,border,roi,boundary=True)
    rs,uf,pp,delta,j5,envelope = bulk
    drs,duf,dpp,ddelta,j1,envelope1 = boundary
    n,pn,d,dt = [weighted_norm(v,w) for v in (rs,pp,rs-uf,pp-uf)]
    actual,actual1 = [weighted_norm(v,w) for v in (delta,ddelta)]
    jref = j5_moments(case,x,y)
    jerr = weighted_norm(j5-jref,w)/weighted_norm(jref,w)
    b0 = float(case.constant*abs(case.pref)*weighted_norm(jref,w))
    b1 = float(case.constant*abs(case.pref)*weighted_norm(j1,w))
    e,eh,practical = d/n,dt/n,dt/pn
    pbound = b0/(pn-b0)*(1+practical) if b0<pn else None
    se = sh = bs = None
    if min(n,d,dt)>1e-14:
        np1 = norm_derivative(rs,drs,w)
        se = norm_derivative(rs-uf,drs-duf,w)/n-d*np1/n**2
        sh = norm_derivative(pp-uf,dpp-duf,w)/n-dt*np1/n**2
        bs = b1/n+2*b0*weighted_norm(dpp-duf,w)/(n*min(d,dt))+b0*weighted_norm(drs,w)/n**2
    sdiff = None if se is None else abs(se-sh)
    numeric_slack = 1e-12  # flag comparisons with less margin than this separately
    result = {**case.metadata(),"aperture_order":order,"roi_order":roi,"boundary_order":border,
              "actual_delta_norm":actual,"B0":b0,"B0_to_actual":ratio(b0,actual),
              "B0_satisfied":bool(actual<=b0),"B0_margin":b0-actual,"normalized_delta":actual/n,
              "actual_derivative_norm":actual1,"B1":b1,"B1_to_actual":ratio(b1,actual1),
              "B1_satisfied":bool(actual1<=b1),"B1_margin":b1-actual1,
              "J5_direct_norm":weighted_norm(j5,w),"J5_gamma_norm":weighted_norm(jref,w),
              "J5_relative_field_difference":jerr,"N_RS":n,"N_PP":pn,"D":d,"D_tilde":dt,
              "E":e,"E_hat_PP":eh,"E_prediction_difference":abs(e-eh),
              "E_prediction_bound":b0/n,"E_bound_satisfied":bool(abs(e-eh)<=b0/n),
              "E_practical":practical,"practical_applicable":b0<pn,"practical_bound":pbound,
              "practical_bound_satisfied":None if pbound is None else bool(abs(e-practical)<=pbound),
              "E_RS_interval_low":max(0.,eh-b0/n),"E_RS_interval_high":eh+b0/n,
              "S_E":se,"S_E_hat_PP":sh,"sensitivity_difference":sdiff,"B_S":bs,
              "BS_applicable":bs is not None,"BS_satisfied":None if bs is None else bool(sdiff<=bs),
              "B_S_to_actual":None if bs is None else ratio(bs,sdiff),
              "B0_pointwise_abs_kernel_integral_norm":weighted_norm(envelope,w),
              "B1_pointwise_abs_kernel_integral_norm":weighted_norm(envelope1,w),
              "B0_cancellation_factor":ratio(weighted_norm(envelope,w),actual),
              "B1_cancellation_factor":ratio(weighted_norm(envelope1,w),actual1),
              "stable_kernel_subtraction_difference":weighted_norm(rs-pp-delta,w),
              "stable_boundary_subtraction_difference":weighted_norm(drs-dpp-ddelta,w),
              "B0_or_B1_margin_near_roundoff":min(b0-actual,b1-actual1)<=numeric_slack}
    return result,bulk,boundary


def crosscheck(case,levels,result,boundary):
    order,roi,_ = levels
    _,_,w = roi_rule(roi,case.ar)
    target = boundary[3]
    previous = extrapolated = None
    rows = []
    for h in HS:
        mode, points, signs = stencil(case.p,h)
        samples = [fd_field(Case(p,case.nf,case.theta,case.ar),order,roi) for p in points]
        derivative = sum(c*(v-samples[0]) for c,v in zip(signs[1:],samples[1:]))/(2*h)
        rich = (4*derivative-previous)/3 if previous is not None else None
        row = {**case.metadata(),"h":h,"stencil":mode,
               "aperture_order":order,"roi_order":roi,"boundary_order":levels[2],
               "shape_derivative_norm":result["actual_derivative_norm"],
               "finite_difference_norm":weighted_norm(derivative,w),
               "finite_difference_vs_analytic_l2":weighted_norm(derivative-target,w),
               "richardson_vs_analytic_l2":None if rich is None else weighted_norm(rich-target,w),
               "richardson_change_l2":None if extrapolated is None else weighted_norm(rich-extrapolated,w)}
        rows.append(row)
        previous,extrapolated = derivative,rich
    return rows


def high_precision_probe():
    """Audit arithmetic of the p=12 FD independently at the on-axis point."""
    import mpmath as m

    with m.workdps(50):
        case = Case(12.,10.,.3)
        z, lam = m.mpf(100)/3,m.mpf(3)/10
        k = 2*m.pi*z/lam
        r,w = unit_legendre(48)
        nodes = [(m.mpf(float(x)),m.mpf(float(v))) for x,v in zip(r,w)]

        def integral(p):
            a = m.sqrt(100*m.pi*m.gamma(1+2/p)/(4*m.gamma(1+1/p)**2))
            terms = []
            for t,wt in nodes:
                edge = a/(1+t**p)**(1/p)
                for rad,wr in nodes:
                    q = (rad*edge/z)**2*(1+t*t)
                    root = m.sqrt(1+q)
                    pp = (q/2+q*q/4)/(1+3*q/4+q*q/16)
                    amp_pp = 1/(1+q)+1j/k*(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48)
                    rs = (1/root**2+1j/(k*root**3))*m.exp(1j*k*(root-1))
                    terms.append((rs-amp_pp*m.exp(1j*k*pp))*wt*wr*rad*edge**2)
            return 8*m.fsum(terms)/(1j*lam*z)

        h = m.mpf("0.0025")
        center,left,far = (integral(12-j*h) for j in range(3))
        derivative = (3*center-4*left+far)/(2*h)
        x=np.array([0.])
        double = []
        for j in range(3):
            p=12-j*float(h)
            xa,ya,w = geometry(p,1,48)
            d = kernels((xa*xa+ya*ya)/case.z**2,case.k)[3]
            double.append((d @ w)*case.pref)
        double_fd = (3*double[0]-4*double[1]+double[2])/(2*float(h))
        return {"digits":50,"p":12,"x":float(x[0]),"y":0,"h":float(h),"aperture_order":48,
                "fd_real":float(derivative.real),"fd_imag":float(derivative.imag),
                "absolute_difference_vs_float64":float(abs(complex(derivative)-double_fd)),
                "scope":"Arithmetic check at the same fixed spatial quadrature; not a spatial error bound."}


def comparisons(case,coarse,fine,level):
    old,of,ob = coarse
    new,nf,nb = fine
    # Fixed probe grid comparison when the aperture/boundary order changes;
    # for ROI refinement compare each field norm on its own normalized rule.
    w = roi_rule(level[1],case.ar)[2]
    w0 = roi_rule(old["roi_order"],case.ar)[2]
    row = {**case.metadata(),"aperture_order":level[0],"roi_order":level[1],
           "boundary_order":level[2],"previous_aperture_order":old["aperture_order"],
           "previous_roi_order":old["roi_order"],"previous_boundary_order":old["boundary_order"]}
    for index,name in enumerate(("RS_I","Fresnel","PP","delta","J5")):
        n = weighted_norm(nf[index],w)
        change = (weighted_norm(nf[index]-of[index],w) if w.size==w0.size
                  else abs(n-weighted_norm(of[index],w0)))
        row[f"{name}_relative_change"] = change/max(n,1e-14)
    for index,name in enumerate(("RS_I_derivative","Fresnel_derivative","PP_derivative","delta_derivative")):
        n = weighted_norm(nb[index],w)
        change = (weighted_norm(nb[index]-ob[index],w) if w.size==w0.size
                  else abs(n-weighted_norm(ob[index],w0)))
        row[f"{name}_relative_change"] = change/max(n,1e-14)
    for key in ("B0","B1","actual_delta_norm","actual_derivative_norm"):
        row[f"{key}_relative_change"] = abs(new[key]-old[key])/max(abs(new[key]),1e-14)
    return row


def plot(out,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['p']:g}/{r['nf']:g}/{r['theta']:g}/{r['ar']:g}" for r in rows]
    for name,actual,bound in (("B0","actual_delta_norm","B0"),("B1","actual_derivative_norm","B1"),
                              ("sensitivity_bound","sensitivity_difference","B_S")):
        fig,ax = plt.subplots(figsize=(8,4),layout="constrained")
        for key,title in ((actual,"Observed"),(bound,"Theoretical bound")):
            points = [v if v is not None and v>0 else np.nan for v in (r[key] for r in rows)]
            ax.semilogy(points,"o-",label=title)
        if len(labels)<=15:
            ax.set(xticks=range(len(labels)),xticklabels=labels)
            ax.tick_params(axis="x",labelrotation=30)
        ax.set(xlabel="p / NF / theta / AR" if len(labels)<=15 else "Case index",title=name)
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(out/f"{name}_comparison.png",dpi=170)
        plt.close(fig)
    from matplotlib.colors import LogNorm
    quantities = ("B0_to_actual","B1_to_actual","B_S_to_actual")
    matrix = np.array([[r[key] if r[key] is not None else np.nan for r in rows] for key in quantities],float)
    fig,ax = plt.subplots(figsize=(8,3.6),layout="constrained")
    heat = ax.imshow(matrix,aspect="auto",norm=LogNorm(vmin=1,vmax=max(2.,np.nanmax(matrix))))
    ax.set(yticks=range(3),yticklabels=["B0 / error","B1 / derivative error","BS / sensitivity error"])
    if len(labels)<=15:
        ax.set(xticks=range(len(labels)),xticklabels=labels)
        ax.tick_params(axis="x",labelrotation=30)
    ax.set_xlabel("p / NF / theta / AR" if len(labels)<=15 else "Case index")
    fig.colorbar(heat,ax=ax,label="Bound / observed")
    fig.savefig(out/"bound_tightness_heatmap.png",dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--quick",action="store_true")
    scope.add_argument("--all",action="store_true",help="Select the 96-case grid plus four asymmetric cases")
    scope.add_argument("--asymmetric-only",action="store_true",help="Select one AR=2 and one AR=4 audit")
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    quick = args.quick or not (args.run or args.all or args.asymmetric_only)
    if args.asymmetric_only:
        cases = [Case(12.,1.,.1,ar) for ar in (2.,4.)]
    elif quick:
        cases = [Case(p,10,.3) for p in (2.,8.,12.)]
    else:
        cases = ([Case(p,n,t) for p in (1.5,2.,3.,4.,8.,12.) for n in (.3,1.,3.,10.)
                  for t in (.05,.1,.2,.3)] if args.all else
                 [Case(p,n,t) for p in (1.5,2.,3.,4.,8.,12.)
                  for n,t in ((.3,.05),(1.,.1),(3.,.2),(10.,.3))])
        cases += [Case(p,10,.3,ar) for ar in (2.,4.) for p in (2.,12.)]
    pairs = 8*sum(a*a*r*r for a,r,_ in LEVELS)+8*48**2*32**2*(2*len(HS)+1)
    print(f"{len(cases)} cases; approx. {len(cases)*pairs:,} aperture/ROI pairs before memoization; "
          "batch array budget ~32 MiB (excludes overhead and exceptional retries).",flush=True)
    if args.plan or not (args.run or args.quick):
        print("Plan only. No integrations performed.")
        return
    proof = symbolic_proof()
    label = ("error_bound_quick" if quick else "error_bound_all" if args.all else
             "error_bound_asymmetric" if args.asymmetric_only else "error_bound_representative")
    out = args.out or Path(__file__).resolve().parent/label
    try:
        out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new --out directory.")
    results, convergence, fd_rows = [],[],[]
    bool_keys = ("B0_satisfied","B1_satisfied","E_bound_satisfied","practical_bound_satisfied","BS_satisfied")
    for case in cases:
        print(f"p={case.p:g}, NF={case.nf:g}, theta={case.theta:g}, AR={case.ar:g}",flush=True)
        previous = None
        for level in LEVELS:
            current = evaluate(case,level)
            if previous is not None:
                convergence.append(comparisons(case,previous,current,level))
            previous = current
        result,_,boundary = current
        if any(result[k] is False for k in bool_keys):
            retry = (80,48,128)
            current = evaluate(case,retry)
            convergence.append(comparisons(case,previous,current,retry))
            result,_,boundary = current
        result["quadrature_last_step_pass"] = bool(
            max(v for row in convergence[-2:] for k,v in row.items()
                if k.endswith("_relative_change"))<=REL_TOL
            and result["J5_relative_field_difference"]<=REL_TOL
        )
        # Same ROI and boundary samples for FD comparison; volume order 48
        # avoids unnecessary repeats while separately testing volume convergence.
        if boundary.shape[1] != 32**2:
            boundary = integrate(case,96,32,boundary=True)
        steps = crosscheck(case,(48,32,96),result,boundary)
        fd_rows.extend(steps)
        relative = steps[-1]["richardson_vs_analytic_l2"]/max(result["actual_derivative_norm"],1e-14)
        result["fd_richardson_final_relative_l2"] = relative
        result["fd_richardson_check_pass"] = relative<=REL_TOL
        results.append(result)
        if any(result[k] is False for k in bool_keys):
            print("WARNING: bound check failed after refinement; halting further cases.",flush=True)
            break
    groups = {
        "bound_B0_validation": ("actual_delta_norm","B0","B0_to_actual","B0_satisfied","B0_margin",
                                "normalized_delta","J5_direct_norm","J5_gamma_norm","J5_relative_field_difference",
                                "B0_cancellation_factor","E","E_hat_PP","E_prediction_difference",
                                "E_prediction_bound","E_bound_satisfied","E_practical","practical_applicable",
                                "practical_bound","practical_bound_satisfied","E_RS_interval_low","E_RS_interval_high"),
        "bound_B1_validation": ("actual_derivative_norm","B1","B1_to_actual","B1_satisfied","B1_margin",
                                "B1_cancellation_factor","fd_richardson_final_relative_l2",
                                "fd_richardson_check_pass"),
        "sensitivity_bound_validation": ("N_RS","D","D_tilde","S_E","S_E_hat_PP","sensitivity_difference",
                                         "B_S","BS_applicable","BS_satisfied","B_S_to_actual"),
    }
    for name,keys in groups.items():
        write_csv(out/f"{name}.csv",[
            {**{k:r[k] for k in ("p","nf","theta","ar","z","wavelength","K","C_K",
                                  "aperture_order","roi_order","boundary_order","quadrature_last_step_pass")},
             **{k:r[k] for k in keys}} for r in results
        ])
    write_csv(out/"shape_derivative_crosscheck.csv",fd_rows)
    write_csv(out/"bound_convergence.csv",convergence)
    plot(out,results)
    report = {
        "symbolic_audit":proof,"quick":quick,"planned_cases":len(cases),"completed_cases":len(results),
        "all_applicable_bounds_pass":all(r[k] is not False for r in results for k in bool_keys),
        "all_numerical_checks_pass":all(r["quadrature_last_step_pass"] and r["fd_richardson_check_pass"]
                                       for r in results),
        "quadrature_relative_tolerance":REL_TOL,
        "physical_units":{"radius":RADIUS,"area":AREA,"roi_radius":ROI,
                          "wavelength_z_and_axes":"common length unit","field":"unit incident amplitude",
                          "q_and_p":"dimensionless","roi":"uniform disk measure normalized to 1"},
        "high_precision_fd_probe":high_precision_probe() if quick and len(results)==3 else None,
        "results":results,
        "tightness_interpretation":[
            "The kernel-envelope integral keeps |Delta F| but discards phase cancellation; "
            "its ratio to the actual field/shape error measures cancellation separately from the q^5 envelope.",
            "For improving the bound, use exact t-dependent remainder denominators and bound "
            "the oscillatory integral with phase-aware estimates; these are directions, not validated replacements.",
            "A near-zero observed sensitivity discrepancy has no useful tightness ratio.",
            "Symbolic bounds are global in q; numerical results apply only to the sampled cases.",
        ],
    }
    (out/"bound_validation_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                                                    encoding="utf-8")
    print(f"Saved to {out.resolve()}")


if __name__=="__main__":
    main()
