"""Separate quadrature convergence from NF/theta applicability of the moment model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from fresnel_moment_sensitivity import (
    AREA, BATCH, FRESNEL_NUMBERS, NEAR_ZERO, PHASE_LIMIT, RADIUS, ROI, SHAPES,
    coefficients, integrate, norm, observation_nodes, phase_limits, save_csv,
)


ROI_ORDERS = (8, 12, 16, 20, 24, 32)
APERTURE_ORDERS = (32, 48, 64, 96, 128, 160)
THETAS = (0.02, 0.05, 0.1, 0.2, 0.3)
ERRORS = ("E", "E_hat", "E_res")
Q_WARNING = 0.1
H = 0.005


@dataclass(frozen=True)
class Condition:
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
    def K(self):
        return 2*np.pi*self.z/self.wavelength


# The original NF=10 run kept wavelength=1, so theta=1 belongs to its
# convergence stress test, outside the new theta<=0.3 applicability grid.
STRESS = Condition(10., 1.)
STRESS_P = 1.5


def verify_conditions(nfs, thetas):
    for nf in nfs:
        for theta in thetas:
            c = Condition(nf, theta)
            np.testing.assert_allclose(RADIUS**2/(c.wavelength*c.z), nf, rtol=1e-14)
            np.testing.assert_allclose(RADIUS/c.z, theta, rtol=1e-14)
    np.testing.assert_allclose([STRESS.wavelength, STRESS.z], [1, 10], rtol=1e-14)
    # For the circular axis case, check absolute scaling at a non-unit wavelength.
    c = Condition(3., 0.2)
    x = y = np.zeros(1)
    values, area = integrate(2., c, 32, x, y)
    exact_rs = 1 - c.z/np.hypot(c.z, RADIUS)*np.exp(
        1j*c.K*(RADIUS/c.z)**2/(np.hypot(1, RADIUS/c.z)+1)
    )
    exact_fr = -np.expm1(1j*np.pi*c.nf)
    np.testing.assert_allclose(values[:2, 0]/(1j*c.wavelength*c.z),
                               [exact_rs, exact_fr], atol=2e-11)
    np.testing.assert_allclose(area, AREA, rtol=1e-12)


def evaluator():
    @lru_cache(maxsize=1024)
    def evaluate(p, c, aperture_order, roi_order):
        x, y, w = observation_nodes(roi_order)
        (rs_integral, m0, m2, m4), area = integrate(p, c, aperture_order, x, y)
        factor = 1/(1j*c.wavelength*c.z)
        rs, fr = factor*rs_integral, factor*m0
        c0, c1, c2 = coefficients(c.K)
        correction = factor*((c0-1)*m0 + c1*m2 + c2*m4)
        denominator = norm(rs, w)
        e = norm(rs-fr, w)/denominator
        prediction = norm(correction, w)/denominator
        residual = norm(rs-fr-correction, w)/denominator
        if abs(e-prediction) > residual + 1e-13:
            raise RuntimeError("Common-denominator field-error bound failed")
        return {
            "E": e, "E_hat": prediction, "E_res": residual,
            "E_hat_legacy": norm(correction, w)/norm(fr+correction, w),
            "area_relative_error": abs(area/AREA-1),
        }

    return evaluate


def context(p, c, aperture_order, roi_order):
    phases = phase_limits(p, c)
    q_warning = phases["q_max"] > Q_WARNING
    phase_warning = phases["phase_indicator_exceeds_pi_over_8"]
    warnings = []
    if q_warning:
        warnings.append("q_max > 0.1")
    if phase_warning:
        warnings.append("quartic phase > pi/8")
    return {
        "p": p, "nf": c.nf, "theta_ap": c.theta, "ar": c.ar,
        "radius": RADIUS, "area": float(AREA), "roi_radius": ROI,
        "wavelength": c.wavelength, "z": c.z, "K": c.K,
        "aperture_order": aperture_order, "roi_order": roi_order,
        **phases, "q_warning": bool(q_warning), "taylor_warning": bool(warnings),
        "warning_reason": "; ".join(warnings),
    }


def convergence(evaluate, aperture_orders, roi_orders):
    finest_aperture, finest_roi = aperture_orders[-1], roi_orders[-1]
    reference = evaluate(STRESS_P, STRESS, finest_aperture, finest_roi)
    studies = {}
    for kind, settings in (
        ("roi", [(finest_aperture, roi) for roi in roi_orders]),
        ("aperture", [(order, finest_roi) for order in aperture_orders]),
    ):
        rows = []
        previous = None
        for order, roi in settings:
            print(f"{kind} convergence: aperture={order}, ROI={roi}", flush=True)
            value = evaluate(STRESS_P, STRESS, order, roi)
            row = {**context(STRESS_P, STRESS, order, roi), **value}
            for key in ERRORS:
                row[f"{key}_abs_change_vs_finest"] = abs(value[key]-reference[key])
                row[f"{key}_abs_change_vs_previous"] = (
                    None if previous is None else abs(value[key]-previous[key])
                )
            rows.append(row)
            previous = value
        studies[kind] = rows
    return studies


def sensitivity(evaluate, p, c, order, roi):
    at = lambda value: evaluate(round(value, 10), c, order, roi)
    center = at(p)
    side = 1 if p == SHAPES[0] else -1 if p == SHAPES[-1] else 0
    if side:
        near, far = at(p+side*H), at(p+2*side*H)
        slopes = [side*(4*(near[key]-center[key])-(far[key]-center[key]))/(2*H)
                  for key in ERRORS[:2]]
    else:
        left, right = at(p-H), at(p+H)
        slopes = [(right[key]-left[key])/(2*H) for key in ERRORS[:2]]
    actual, predicted = slopes
    delta = abs(actual-predicted)
    return {
        **context(p, c, order, roi), **center,
        "error_prediction_rel_error": abs(center["E"]-center["E_hat"])/center["E"],
        "fd_h": H, "S_E": actual, "S_E_hat": predicted, "sensitivity_abs_error": delta,
        "sensitivity_rel_error": None if abs(actual) < NEAR_ZERO else delta/abs(actual),
        "sensitivity_sign_agrees": None if min(abs(actual), abs(predicted)) < NEAR_ZERO
        else bool(np.sign(actual) == np.sign(predicted)),
    }


def heatmap(out, name, groups, nfs, thetas, panels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.8*len(panels), 4.7),
                             layout="constrained", squeeze=False)
    for ax, (field, label) in zip(axes[0], panels):
        matrix = np.empty((len(nfs), len(thetas)))
        flagged = []
        for i, nf in enumerate(nfs):
            for j, theta in enumerate(thetas):
                rows = groups[nf, theta]
                matrix[i, j] = max((abs(r[field]) for r in rows if r[field] is not None), default=np.nan)
                if any(r["taylor_warning"] for r in rows):
                    flagged.append((i, j))
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, shrink=0.75)
        for i, j in flagged:
            ax.plot(j, i, "s", markersize=12, markerfacecolor="none", markeredgecolor="#ffb75b")
        ax.set(xticks=range(len(thetas)), xticklabels=[f"{t:g}" for t in thetas],
               yticks=range(len(nfs)), yticklabels=[f"{n:g}" for n in nfs],
               xlabel="Aperture angle parameter theta", ylabel="Fresnel number", title=label)
    fig.suptitle(title + "\nOutline indicates a Taylor warning for at least one sampled p")
    fig.savefig(out / name, dpi=170)
    plt.close(fig)


def print_plan(aperture_orders, roi_orders, nfs, thetas, shapes, sweep_order, sweep_roi, args):
    convergence_pairs = 4*(aperture_orders[-1]**2*sum(r*r for r in roi_orders)
                           + roi_orders[-1]**2*sum(a*a for a in aperture_orders[:-1]))
    sweep_pairs = 12*sweep_order**2*sweep_roi**2*len(nfs)*len(thetas)*len(shapes)
    if args.sweep_only:
        convergence_pairs = 0
    if args.convergence_only:
        sweep_pairs = 0
    peak_order = max(sweep_order if sweep_pairs else 0,
                     aperture_orders[-1] if convergence_pairs else 0)
    cost = {"convergence_pairs": convergence_pairs, "sweep_pairs_upper": sweep_pairs,
            "quadrature_workspace_budget_mib": BATCH*4*peak_order**2*128/2**20}
    print(f"Stress test: NF=10, theta=1, AR=1, p=1.5, wavelength=1, z=10.")
    print(f"ROI orders {roi_orders}; aperture orders {aperture_orders}.")
    print(f"Sweep: NF={nfs}, theta={thetas}, p={shapes}, orders={sweep_order}/{sweep_roi}.")
    print(f"Point pairs: convergence {convergence_pairs:,}; sweep up to {sweep_pairs:,}.")
    print(f"Batch array budget ~ {cost['quadrature_workspace_budget_mib']:.1f} MiB "
          "(excludes interpreter and plotting).", flush=True)
    return cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Print estimates without integrating")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Run both studies")
    mode.add_argument("--convergence-only", action="store_true")
    mode.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use reduced quadrature and a subset of the sweep")
    parser.add_argument("--sweep-aperture-order", type=int, help="Override the 2D sweep aperture order")
    parser.add_argument("--sweep-roi-order", type=int, help="Override the 2D sweep observation order")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    aperture_orders = (32, 48, 64) if args.quick else APERTURE_ORDERS
    roi_orders = (8, 12, 16) if args.quick else ROI_ORDERS
    nfs = (0.1, 1., 10.) if args.quick else FRESNEL_NUMBERS
    thetas = (0.02, 0.1, 0.3) if args.quick else THETAS
    shapes = (1., 1.5, 2., 12.) if args.quick else SHAPES
    sweep_order, sweep_roi = (32, 12) if args.quick else (64, 20)
    if args.sweep_aperture_order is not None:
        sweep_order = args.sweep_aperture_order
    if args.sweep_roi_order is not None:
        sweep_roi = args.sweep_roi_order
    if min(sweep_order, sweep_roi) < 2:
        parser.error("Sweep quadrature orders must be at least two")
    cost = print_plan(aperture_orders, roi_orders, nfs, thetas, shapes, sweep_order, sweep_roi, args)
    if args.plan or not (args.run or args.convergence_only or args.sweep_only or args.quick):
        print("Plan only. No field integrations performed.")
        return
    verify_conditions(nfs, thetas)
    evaluate = evaluator()
    out = args.out or Path("moment_sensitivity_v2_quick" if args.quick else "moment_sensitivity_v2_results")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "quick": args.quick, "plan": cost, "meep_executed": False,
        "fixed_radius": RADIUS, "fixed_area": float(AREA), "fixed_roi_radius": ROI, "ar": 1,
        "parameterization": "z=a_eff/theta_ap; wavelength=a_eff*theta_ap/NF",
        "notes": [
            "V1's fixed-wavelength NF sweep varies theta_ap simultaneously. Its NF=10 "
            "stress case has theta_ap=1 and sits outside this independent theta grid.",
            "Independent NF/theta coordinates are obtained by varying physical wavelength and distance.",
            "All errors share the RS-I denominator except the explicitly labeled legacy column.",
            "Heatmaps aggregate maxima over sampled p, so different panels can peak at different p.",
            "Warnings use q_max>0.1 or leading quartic phase>pi/8 as heuristic indicators, not fitted validity limits.",
            "Each convergence sequence fixes the other integration order at the largest tested value.",
        ],
        "thresholds": {"q_max": Q_WARNING, "quartic_phase_rad": float(PHASE_LIMIT),
                       "near_zero_slope_abs": NEAR_ZERO},
    }
    if not args.sweep_only:
        for kind, rows in convergence(evaluate, aperture_orders, roi_orders).items():
            save_csv(out / f"{kind}_quadrature_convergence.csv", rows)
            report[f"{kind}_convergence"] = {
                "finest": rows[-1],
                "last_step_absolute_changes": {key: rows[-1][f"{key}_abs_change_vs_previous"]
                                               for key in ERRORS},
                "last_step_relative_changes": {
                    key: rows[-1][f"{key}_abs_change_vs_previous"]/abs(rows[-1][key])
                    for key in ERRORS
                },
            }
    if not args.convergence_only:
        groups = {}
        for nf in nfs:
            for theta in thetas:
                print(f"Sweep: NF={nf:g}, theta={theta:g}", flush=True)
                c = Condition(nf, theta)
                groups[nf, theta] = [sensitivity(evaluate, p, c, sweep_order, sweep_roi) for p in shapes]
        rows = [row for group in groups.values() for row in group]
        save_csv(out / "nf_theta_2d_sweep.csv", rows)
        heatmap(out, "nf_theta_error_heatmap.png", groups, nfs, thetas,
                (("E", "Max direct error"), ("E_hat", "Max predicted error"),
                 ("E_res", "Max Taylor residual")), "NF–theta field errors")
        heatmap(out, "nf_theta_sensitivity_heatmap.png", groups, nfs, thetas,
                (("S_E", "Max |direct slope|"), ("S_E_hat", "Max |predicted slope|"),
                 ("sensitivity_abs_error", "Max absolute slope discrepancy")), "NF–theta shape sensitivities")
        report["sweep"] = {
            "samples": len(rows), "warned_samples": sum(r["taylor_warning"] for r in rows),
            "max_area_relative_error": max(r["area_relative_error"] for r in rows),
            "worst_magnitude_prediction_case": max(rows, key=lambda r: r["error_prediction_rel_error"]),
            "worst_sensitivity_case": max(rows, key=lambda r: r["sensitivity_abs_error"]),
            "opposite_sign_samples": sum(r["sensitivity_sign_agrees"] is False for r in rows),
        }
    (out / "applicability_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
