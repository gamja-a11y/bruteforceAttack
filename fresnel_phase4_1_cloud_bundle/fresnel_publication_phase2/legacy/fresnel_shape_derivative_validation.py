"""Recompute and independently audit superellipse Fresnel error derivatives."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import beta, digamma, gammaln

from fresnel_moment_sensitivity import coefficients
from fresnel_quadrature_validation import (
    AREA, K, NF, PREF, RADIUS, ROI_RADIUS, THETA, WAVELENGTH, Z, unit_legendre,
    weighted_norm, write_csv,
)
from fresnel_roi_convergence import polar_rule, verify_roi


AR = 1.
CORE = (1., 1.5, 2., 3., 4., 6., 8., 12.)
NEAR = (11.98, 11.99, 11.995, 12.)
STEPS = (.02, .01, .005, .0025, .00125)
APERTURE = (24, 32, 48, 64)
BASE = (64, 32, 32)
ERRORS = ("E", "E_hat", "E_res")
SLOPES = ("S_E", "S_E_hat")
BATCH = 6
REL_TOL = .01
ABS_TOL = 1e-7
ERROR_ABS_TOL = 1e-8


def half_axis(p):
    factor = 4*np.exp(2*gammaln(1+1/p)-gammaln(1+2/p))
    return float(np.sqrt(AREA/factor))


def aperture(p, order):
    a = half_axis(p)
    r, wr = unit_legendre(order)
    t, wt = unit_legendre(order)
    edge = a*(1+t**p)**(-1/p)
    # x=r*edge, y=t*x; the positive determinant is r*edge**2.
    x = np.outer(r, edge).ravel()
    y = np.outer(r, t*edge).ravel()
    w = np.tile(np.outer(wr*r, wt*edge**2).ravel(), 8)
    xs = np.concatenate((x,-x,-x,x,y,-y,-y,y))
    ys = np.concatenate((y,y,-y,-y,x,x,-x,-x))
    second = 4*a**4*beta(3/p,1+1/p)/p
    area_relative = abs(w.sum()/AREA-1)
    moment_relative = max(abs(np.dot(w, v*v)/second-1) for v in (xs, ys))
    region = np.max((abs(xs)/a)**p+(abs(ys)/a)**p)
    if min(w) <= 0 or region > 1+2e-13 or max(area_relative, moment_relative) > 5e-6:
        raise RuntimeError(f"Invalid aperture quadrature at p={p}, order={order}")
    return xs, ys, w, {
        "semi_axis": a, "area_relative_error": float(area_relative),
        "jacobian_second_moment_relative_error": float(moment_relative),
    }


@lru_cache(maxsize=None)
def observation(radial, angular):
    if AR != 1:
        raise ValueError("Octant quadrature requires AR=1")
    rule = polar_rule(radial, angular)
    verify_roi(*rule)
    return rule


@lru_cache(maxsize=None)
def evaluate(p, order, radial, angular):
    xa, ya, wa, audit = aperture(p, order)
    x, y, w = observation(radial, angular)
    amplitudes = np.empty((3, x.size), dtype=complex)
    c0, c1, c2 = coefficients(K)
    for start in range(0, x.size, BATCH):
        s = slice(start, start+BATCH)
        q = ((x[s,None]-xa)**2+(y[s,None]-ya)**2)/Z**2
        root = np.sqrt(1+q)
        rs = np.exp(1j*K*q/(root+1))*(1+1j/(K*root))/(1+q)
        fresnel = np.exp(.5j*K*q)*wa
        qm = q*fresnel
        m0, m2, m4 = (np.sum(items,axis=1) for items in (fresnel,qm,q*qm))
        amplitudes[:,s] = PREF*np.array([
            np.sum(wa*rs,axis=1), m0, (c0-1)*m0+c1*m2+c2*m4,
        ])
    rs, fr, correction = amplitudes
    denominator = weighted_norm(rs,w)
    results = dict(zip(ERRORS, (
        weighted_norm(rs-fr,w)/denominator, weighted_norm(correction,w)/denominator,
        weighted_norm(rs-fr-correction,w)/denominator,
    )))
    return {**results, **audit}


def stencil(p, h):
    if p-h < 1:
        mode, offsets, signs = "forward_2", (0,h,2*h), (0,4,-1)
    elif p+h > 12+1e-13:
        mode, offsets, signs = "backward_2", (0,-h,-2*h), (0,-4,1)
    else:
        mode, offsets, signs = "centered_2", (0,-h,h), (0,-1,1)
    return mode, tuple(round(p+d, 10) for d in offsets), signs


def difference(p, grid, h):
    mode, points, coeffs = stencil(p, h)
    center, *other = [evaluate(q,*grid) for q in points]
    slopes = [sum(c*(v[key]-center[key]) for c,v in zip(coeffs[1:],other))/(2*h)
              for key in ERRORS[:2]]
    absolute = abs(slopes[0]-slopes[1])
    return {
        "p": p, "aperture_order": grid[0], "radial_order": grid[1], "angular_order": grid[2],
        "h": h, "stencil": mode, **center, **dict(zip(SLOPES,slopes)),
        "sensitivity_abs_difference": absolute,
        "sensitivity_relative_difference": absolute/abs(slopes[0]) if abs(slopes[0]) > ABS_TOL else None,
        "sensitivity_sign_agrees": bool(np.sign(slopes[0])==np.sign(slopes[1]))
        if min(map(abs,slopes)) > ABS_TOL else None,
    }


def refinement_rows(p, configs, axis):
    rows = []
    for grid in configs:
        value = difference(p,grid,STEPS[-1])
        row = {"varied": axis, **value}
        if rows:
            compare(row, rows[-1])
        rows.append(row)
    return rows


def compare(row, previous):
    for key in (*ERRORS,*SLOPES):
        delta = abs(row[key]-previous[key])
        row[f"{key}_abs_change"] = delta
        row[f"{key}_rel_change"] = delta/abs(row[key]) if abs(row[key]) > ABS_TOL else None


def passes(row, previous):
    return all(abs(row[k]-previous[k]) <= max(ABS_TOL, REL_TOL*abs(row[k])) for k in SLOPES)


def errors_pass(row, previous):
    return all(abs(row[k]-previous[k]) <= ERROR_ABS_TOL for k in ERRORS)


def fd_study(p):
    rows = []
    for h in STEPS:
        row = difference(p,BASE,h)
        if rows:
            compare(row, rows[-1])
            if row["stencil"] == rows[-1]["stencil"]:
                for key in SLOPES:
                    row[f"{key}_richardson"] = (4*row[key]-rows[-1][key])/3
        rows.append(row)
    return rows


def shape_check():
    p = 12.
    a = half_axis(p)
    analytic = a*(digamma(1+1/p)-digamma(1+2/p))/p**2
    h = STEPS[-1]
    numeric = (3*a-4*half_axis(p-h)+half_axis(p-2*h))/(2*h)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)
    directions = []
    for t in (0., .5, 1.):
        t_power = t**p
        slope = analytic/a+np.log1p(t_power)/p**2
        if t:
            slope -= t_power*np.log(t)/(p*(1+t_power))
        directions.append({"ray_slope_y_over_x": t, "radial_boundary_derivative":
                           float(a*(1+t_power)**(-1/p)*np.hypot(1,t)*slope)})
    return {"semi_axis_derivative_analytic": float(analytic),
            "semi_axis_derivative_backward": float(numeric), "ray_boundary_derivatives": directions}


def grids(roi_max):
    levels = [value for value in (48,64,96) if value <= roi_max]
    return {
        "aperture": [(a,32,32) for a in APERTURE],
        "radial": [BASE, *((64,n,32) for n in levels)],
        "angular": [BASE, *((64,32,n) for n in levels)],
        "joint": [BASE, (64,roi_max,roi_max)],
    }


def cost_plan(ps, settings):
    evaluations = {(q,*BASE) for p in ps for h in STEPS for q in stencil(p,h)[1]}
    for configurations in settings.values():
        evaluations |= {(q,*g) for g in configurations for p in ps for q in stencil(p,STEPS[-1])[1]}
    pairs = sum(8*a*a*r*t for _,a,r,t in evaluations)
    print(f"p={ps}; {len(evaluations)} unique field integrations; {pairs:,} point pairs; "
          "approximately 30 MiB for batched arrays, excluding library overhead.",flush=True)
    return {"unique_field_integrations": len(evaluations), "point_pairs": pairs}


def plots(out, results, fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, keys, ylabel in (
        ("shape_error_comparison", ERRORS, "Relative field error"),
        ("shape_sensitivity_comparison", SLOPES, "Shape sensitivity"),
    ):
        fig, ax = plt.subplots(figsize=(7,4),layout="constrained")
        for k in keys:
            ax.plot([r["p"] for r in results], [np.nan if r[k] is None else r[k] for r in results],
                    "o--" if k in ("E_hat","S_E_hat") else "o-", label=k)
        ax.set(xlabel="Shape exponent p",ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(out/f"{filename}.png",dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,k in zip(axes,SLOPES):
        for p in sorted({r["p"] for r in fd}):
            rows = [r for r in fd if r["p"]==p]
            line, = ax.semilogx([r["h"] for r in rows],[r[k] for r in rows],"o-",label=f"p={p:g}")
            rich = [r for r in rows if f"{k}_richardson" in r]
            ax.semilogx([r["h"] for r in rich],[r[f"{k}_richardson"] for r in rich],
                        "x--",color=line.get_color())
        ax.set(xlabel="Finite-difference step h",title=f"{k} · dashed: Richardson")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/"finite_difference_convergence.png",dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--quick",action="store_true")
    scope.add_argument("--all",action="store_true")
    parser.add_argument("--roi-max",type=int,choices=(48,64,96),default=48)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    ps = NEAR if args.quick else tuple(sorted(set(CORE+NEAR)))
    settings = grids(args.roi_max)
    cost = cost_plan(ps,settings)
    if args.plan or not (args.run or args.quick):
        print("Plan only. No field integrations performed.")
        return
    out = args.out or Path(__file__).resolve().parent/("shape_derivative_quick" if args.quick
                                                        else "shape_derivative_results")
    try:
        out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error("Output already exists. Choose a new --out.")
    np.testing.assert_allclose([RADIUS**2/(WAVELENGTH*Z),RADIUS/Z],[NF,THETA],rtol=1e-14)
    results, fd_rows, ap_rows, roi_rows, audits = [],[],[],[],[]
    for p in ps:
        print(f"p={p:g}: finite differences and independent quadrature checks",flush=True)
        fd = fd_study(p)
        fd_rows.extend(fd)
        rows = {key: refinement_rows(p,configs,key) for key,configs in settings.items()}
        ap_rows.extend(rows["aperture"])
        roi_rows.extend(row for key in ("radial","angular","joint") for row in rows[key])
        final = fd[-1]
        # The published FD uses the verified 32x32 rule. Compare each refined
        # direction and their joint grid directly to that actual FD setting.
        pairs = {"aperture": (rows["aperture"][-1],rows["aperture"][-2])}
        pairs.update({key:(value[-1],final) for key,value in rows.items() if key!="aperture"})
        error_checks = {key:errors_pass(*pair) for key,pair in pairs.items()}
        derivative_checks = {key:passes(*pair) for key,pair in pairs.items()}
        fd_pass = all(passes(a,b) for a,b in zip(fd[-2:],fd[-3:-1]))
        rich = [{key:r.get(f"{key}_richardson") for key in SLOPES} for r in fd[-2:]]
        rich_pass = all(v is not None for r in rich for v in r.values()) and passes(rich[1],rich[0])
        status = all(error_checks.values()) and all(derivative_checks.values()) and fd_pass and rich_pass
        audit = {"p":p, "error_quadrature_checks":error_checks,
                 "derivative_quadrature_checks":derivative_checks,
                 "finite_difference_checks_pass":fd_pass,"richardson_check_pass":rich_pass}
        audits.append(audit)
        if not status:
            print(f"WARNING: p={p:g} has at least one unresolved convergence check",flush=True)
        results.append({**final, "converged":status,
                        "S_E_provisional":final["S_E"],"S_E_hat_provisional":final["S_E_hat"],
                        "S_E":final["S_E"] if status else None,
                        "S_E_hat":final["S_E_hat"] if status else None,
                        "S_E_richardson":rich[-1]["S_E"] if status else None,
                        "S_E_hat_richardson":rich[-1]["S_E_hat"] if status else None})
    for name, rows in (("shape_derivative_results",results),("finite_difference_convergence",fd_rows),
                       ("aperture_convergence",ap_rows),("roi_convergence",roi_rows)):
        write_csv(out/f"{name}.csv",rows)
    plots(out,results,fd_rows)
    report = {
        "quick":args.quick,"cost":cost,"checks":audits,"p12":results[-1],
        "physical_conditions":{"nf":NF,"theta_ap":THETA,"ar":AR,"radius":RADIUS,
                               "area":AREA,"z":Z,"wavelength":WAVELENGTH,"roi_radius":ROI_RADIUS},
        "geometry_at_p12":shape_check(),
        "cache":evaluate.cache_info()._asdict(),
        "tolerances":{"slope_relative":REL_TOL,"slope_absolute_floor":ABS_TOL,
                      "error_quadrature_absolute":ERROR_ABS_TOL},
        "notes":[
            "Only octant-symmetric AR=1 fields are evaluated. Quadrature is evaluated anew for each p.",
            "Separate aperture and ROI checks include the slopes at the smallest h. "
            "The shape reported in results uses the 32x32 ROI only after checking the refined rules.",
            "Richardson is used only for consecutive matching second-order stencils. "
            "An interior point too close to the domain endpoint uses a one-sided stencil for that step.",
            "A persistent direct/predicted slope discrepancy after convergence refers to the scalar moment model.",
        ],
    }
    (out/"shape_derivative_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                                                    encoding="utf-8")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
