"""Independent scalar quadrature audit of Fresnel complex-moment sensitivities."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gammaln, roots_jacobi, roots_legendre


SHAPES = (1., 1.5, 2., 3., 4., 6., 8., 12.)
FRESNEL_NUMBERS = (0.1, 0.3, 1., 3., 10.)
ASPECT_RATIOS = (1., 2., 4.)
ORDERS = (32, 48, 64, 96, 128)
STEPS = (0.02, 0.01, 0.005)
WAVELENGTH, RADIUS, ROI = 1., 10., 5.
AREA = np.pi * RADIUS**2
BATCH = 8
NEAR_ZERO = 1e-7
ERROR_REL_TOL, SLOPE_REL_TOL = 0.05, 0.10
PHASE_LIMIT = np.pi / 8
VALUES = ("E", "E_hat", "E_res", "S_E", "S_E_hat")


@dataclass(frozen=True)
class Condition:
    nf: float = 1.
    ar: float = 1.

    @property
    def z(self):
        return RADIUS**2 / (WAVELENGTH * self.nf)

    @property
    def K(self):
        return 2 * np.pi * self.z / WAVELENGTH


def half_axes(p, ar):
    factor = 4 * np.exp(2 * gammaln(1 + 1/p) - gammaln(1 + 2/p))
    ay = np.sqrt(AREA / (ar * factor))
    return ar * ay, ay


@lru_cache(maxsize=None)
def legendre(order):
    nodes, weights = roots_legendre(order)
    return (nodes + 1) / 2, weights / 2


def aperture_nodes(p, ar, order):
    """Map the unit square to each quadrant; absorb the edge in Jacobi weights."""
    nodes, weights = roots_jacobi(order, 1/p, 0)
    u = (nodes + 1) / 2
    v, wv = legendre(order)
    edge = (-np.expm1(p * np.log(u)))**(1/p)
    ax, ay = half_axes(p, ar)
    x = np.broadcast_to(ax * u[:, None], (order, order)).ravel()
    y = (ay * edge[:, None] * v).ravel()
    jacobian = ax * ay * edge / (1 - u)**(1/p)
    weights = np.outer(weights * jacobian / 2**(1 + 1/p), wv).ravel()
    return (np.concatenate((x, -x, -x, x)), np.concatenate((y, y, -y, -y)),
            np.tile(weights, 4))


@lru_cache(maxsize=None)
def observation_nodes(order):
    u, w = legendre(order)
    # Both fields are even in x and y. Integrating one quadrant suffices.
    angle = np.pi / 2 * u
    x = np.outer(ROI * u, np.cos(angle)).ravel()
    y = np.outer(ROI * u, np.sin(angle)).ravel()
    weights = np.outer(w*u, w).ravel()
    return x, y, weights / weights.sum()


def coefficients(K):
    c0 = 1 + 1j/K
    return c0, -1 - 1.5j/K, 1 + 15j/(8*K) - 1j*K*c0/8


def integrate(p, condition, order, x, y):
    xa, ya, weights = aperture_nodes(p, condition.ar, order)
    K, z = condition.K, condition.z
    values = np.empty((4, x.size), dtype=complex)
    for start in range(0, x.size, BATCH):
        s = slice(start, start + BATCH)
        q = ((x[s, None] - xa)**2 + (y[s, None] - ya)**2) / z**2
        root = np.sqrt(1 + q)
        # Algebraic form of sqrt(1+q)-1 that avoids cancellation at small q.
        rs = np.exp(1j*K*q/(root + 1)) * (1 + 1j/(K*root)) / (1 + q)
        fresnel = np.exp(0.5j*K*q) * weights
        weighted_q = q * fresnel
        values[:, s] = (np.sum(rs * weights, axis=1), np.sum(fresnel, axis=1),
                        np.sum(weighted_q, axis=1), np.sum(q * weighted_q, axis=1))
    return values, float(weights.sum())


def norm(field, weights):
    return float(np.sqrt(np.dot(weights, abs(field)**2)))


def field_metrics(field, reference, weights, label):
    overlap = np.vdot(reference, weights * field)
    offset = np.angle(overlap)
    phase_weights = weights * abs(reference)**2
    phase = np.angle(field * reference.conj())
    aligned_phase = np.angle(np.exp(1j * (phase - offset)))
    reference_norm = norm(reference, weights)
    raw = norm(field-reference, weights) / reference_norm
    return {
        f"{label}_relative_complex_l2": raw,
        f"{label}_raw_relative_l2": raw,
        f"{label}_phase_aligned_relative_l2":
            norm(field*np.exp(-1j*offset)-reference, weights) / reference_norm,
        f"{label}_intensity_l1":
            float(np.dot(weights, abs(abs(field)**2-abs(reference)**2)) / phase_weights.sum()),
        f"{label}_weighted_phase_rms_raw_rad": norm(phase, phase_weights) / reference_norm,
        f"{label}_weighted_phase_rms_aligned_rad":
            norm(aligned_phase, phase_weights) / reference_norm,
    }


def phase_limits(p, condition):
    ax, ay = half_axes(p, condition.ar)
    major, minor = max(ax, ay), min(ax, ay)
    if p > 2:
        power = 2*p/(p-2)
        major *= (1 + (minor/major)**power)**(1/power)
    qmax = ((major + ROI) / condition.z)**2
    quartic = condition.K*qmax*qmax/8
    exact = condition.K*qmax*qmax / (2*(1 + np.sqrt(1 + qmax))**2)
    return {
        "q_max": qmax, "max_omitted_fresnel_quartic_phase_rad": quartic,
        "max_exact_fresnel_phase_defect_rad": float(exact),
        "phase_indicator_exceeds_pi_over_8": bool(quartic > PHASE_LIMIT),
    }


def evaluator():
    @lru_cache(maxsize=512)
    def evaluate(p, condition, order, roi_order):
        x, y, weights = observation_nodes(roi_order)
        (rs_integral, m0, m2, m4), area = integrate(p, condition, order, x, y)
        prefactor = 1 / (1j * WAVELENGTH * condition.z)
        rs, fr = prefactor * rs_integral, prefactor * m0
        c0, c1, c2 = coefficients(condition.K)
        correction = prefactor * ((c0-1)*m0 + c1*m2 + c2*m4)
        predicted = fr + correction
        denominator = norm(rs, weights)
        e = norm(rs-fr, weights) / denominator
        e_hat = norm(correction, weights) / denominator
        residual = norm(rs-predicted, weights) / denominator
        if abs(e - e_hat) > residual + 1e-14:
            raise RuntimeError("The field-error triangle inequality failed")
        return {
            "E": e, "E_hat": e_hat, "E_res": residual,
            "E_hat_legacy": norm(correction, weights) / norm(predicted, weights),
            "area_numeric": area, "area_abs_relative_error": abs(area/AREA-1),
            "direct_error_exceeds_5_percent": e > 0.05,
            **field_metrics(fr, rs, weights, "fresnel_vs_rs"),
            **field_metrics(predicted, rs, weights, "moment_vs_rs"),
            **field_metrics(predicted, fr, weights, "moment_vs_fresnel"),
        }

    return evaluate


def sensitivity(evaluate, p, condition, order, roi_order, h):
    center = evaluate(p, condition, order, roi_order)
    direction = 1 if p == SHAPES[0] else -1 if p == SHAPES[-1] else 0
    if direction:
        near, far = [evaluate(round(p + direction*h*i, 10), condition, order, roi_order)
                     for i in (1, 2)]
        slopes = [direction*(4*(near[k]-center[k]) - (far[k]-center[k]))/(2*h)
                  for k in ("E", "E_hat")]
        stencil = "forward" if direction == 1 else "backward"
    else:
        right = evaluate(round(p+h, 10), condition, order, roi_order)
        left = evaluate(round(p-h, 10), condition, order, roi_order)
        slopes = [(right[k]-left[k])/(2*h) for k in ("E", "E_hat")]
        stencil = "centered"
    actual, estimate = slopes
    absolute = abs(actual-estimate)
    near_zero = abs(actual) < NEAR_ZERO
    relative = None if near_zero else absolute/abs(actual)
    return {
        "p": p, "nf": condition.nf, "ar": condition.ar, "z": condition.z,
        "wavelength": WAVELENGTH, "a_eff": RADIUS, "roi_radius": ROI,
        "theta_ap": RADIUS/condition.z, "area_target": float(AREA),
        "a_x": half_axes(p, condition.ar)[0], "a_y": half_axes(p, condition.ar)[1],
        "order": order, "roi_order": roi_order, "h": h, "stencil": stencil,
        **center, **phase_limits(p, condition),
        "S_E": actual, "S_E_hat": estimate, "sensitivity_abs_error": absolute,
        "sensitivity_rel_error": relative, "sensitivity_near_zero": near_zero,
        "direction_agrees": None if near_zero or abs(estimate) < NEAR_ZERO
        else bool(np.sign(actual) == np.sign(estimate)),
        "error_prediction_rel_error": abs(center["E"]-center["E_hat"])/center["E"],
    }


def independent_checks():
    import sympy as sp
    import fresnel_shape_sensitivity as legacy

    q, K = sp.symbols("q K", real=True, positive=True)
    ratio = sp.exp(sp.I*K*(sp.sqrt(1+q)-1-q/2)) * (1+sp.I/(K*sp.sqrt(1+q))) / (1+q)
    taylor = sp.series(ratio, q, 0, 3).removeO().expand()
    expected = (1+sp.I/K, -1-3*sp.I/(2*K), 1+15*sp.I/(8*K)-sp.I*K*(1+sp.I/K)/8)
    for power, value in enumerate(expected):
        assert sp.simplify(taylor.coeff(q, power)-value) == 0
    numerical = tuple(complex(value.subs(K, Condition().K)) for value in expected)
    np.testing.assert_allclose(coefficients(Condition().K), numerical, rtol=1e-14)
    for nf in (0.1, 1., 10.):
        condition = Condition(nf)
        z, k = condition.z, condition.K/condition.z
        radius = np.hypot(z, RADIUS)
        exact_rs = 1 - z/radius * np.exp(1j*k*RADIUS**2/(radius+z))
        exact_fr = -np.expm1(0.5j*k*RADIUS**2/z)
        axis = np.zeros(1)
        values, area = integrate(2., condition, 48, axis, axis)
        np.testing.assert_allclose(values[:2, 0]/(1j*WAVELENGTH*z),
                                   [exact_rs, exact_fr], atol=2e-11)
        np.testing.assert_allclose(area, AREA, rtol=1e-12)
    comparisons = []
    x, y, weights = observation_nodes(12)
    condition = Condition()
    for p in SHAPES:
        values, _ = integrate(p, condition, 64, x, y)
        new = values.T/(1j*WAVELENGTH*condition.z)
        old, *_ = legacy.fields(p, legacy.Experiment(), 64, x, y)
        differences = [norm(a-b, weights)/norm(b, weights) for a, b in zip(new.T, old.T)]
        old_error = legacy.evaluate(p, legacy.Experiment(), 64, (x, y, weights))
        verify = evaluator()(p, condition, 64, 12)
        comparisons.append({
            "p": p, "rs_fresnel_scaled_m2_m4_relative_l2": differences,
            "direct_E_abs_difference": abs(verify["E"]-old_error["E"]),
            "legacy_E_hat_abs_difference": abs(verify["E_hat_legacy"]-old_error["E_hat"]),
            "common_denominator_E_hat": verify["E_hat"],
        })
    return {
        "sympy_taylor_coefficients": [str(sp.simplify(taylor.coeff(q, i))) for i in range(3)],
        "circular_on_axis_analytic_checks_nf": [0.1, 1., 10.],
        "legacy_field_comparison": comparisons,
        "legacy_field_comparison_within_1e-6": all(
            max(row["rs_fresnel_scaled_m2_m4_relative_l2"]) <= 1e-6 for row in comparisons
        ),
        "legacy_denominator_difference": "Legacy E_hat divides by the predicted field norm; "
        "the primary E_hat here divides by the direct RS-I norm as specified.",
        "diagnostic_source": "Reviewed meep_shape_diagnostic_v2.py; the requested (1) filename was absent.",
        "meep_executed": False,
    }


def convergence(evaluate, shapes, orders, roi_order, condition, *, extra_order=False):
    output, finest = [], []
    for p in shapes:
        levels = (*orders, 160) if p == 1.5 and extra_order else orders
        reference = sensitivity(evaluate, p, condition, levels[-1], roi_order, STEPS[-1])
        finest.append(reference)
        variations = [
            ("aperture_order", order, roi_order, STEPS[-1]) for order in levels
        ] + [
            ("finite_difference_step", levels[-1], roi_order, h) for h in STEPS
        ] + [("roi_order", levels[-1], roi_order+4, STEPS[-1])]
        for factor, order, roi, h in variations:
            row = sensitivity(evaluate, p, condition, order, roi, h)
            output.append({"varied": factor, **row, **{
                f"{key}_abs_change_vs_reference": abs(row[key]-reference[key]) for key in VALUES
            }, "reference_order": levels[-1], "reference_roi_order": roi_order,
                "reference_h": STEPS[-1]})
    return finest, output


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(out, name, groups, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(panels), 1, figsize=(8.5, 3.8*len(panels)),
                                squeeze=False, sharex=True, layout="constrained")
    colors = plt.get_cmap("tab10")
    for ax, panel in zip(axes[:, 0], panels):
        is_error = panel == "error"
        flag_added = False
        direct, predicted = ("E", "E_hat") if is_error else ("S_E", "S_E_hat")
        for index, (label, rows) in enumerate(groups.items()):
            p = [r["p"] for r in rows]
            prefix = f"{label} · " if label else ""
            ax.plot(p, [100*r[direct] for r in rows], "o-", color=colors(index),
                    label=f"{prefix}direct")
            ax.plot(p, [100*r[predicted] for r in rows], "s--", color=colors(index),
                    markerfacecolor="white", label=f"{prefix}moment")
            if is_error and len(groups) == 1:
                ax.plot(p, [100*r["E_res"] for r in rows], "^:", color="0.35", label="Model residual")
            if is_error:
                flagged = [r for r in rows if r["phase_indicator_exceeds_pi_over_8"]
                           or r["direct_error_exceeds_5_percent"]]
                if flagged:
                    ax.scatter([r["p"] for r in flagged], [100*r["E"] for r in flagged],
                               s=110, facecolors="none", edgecolors=colors(index),
                               label=None if flag_added else "Fresnel error or phase above threshold")
                    flag_added = True
        ax.grid(alpha=0.25)
        ax.set_ylabel("Weighted field error (%)" if is_error else "Sensitivity (% per unit p)")
        ax.legend(ncols=2, fontsize=8)
    axes[-1, 0].set_xlabel("Superellipse exponent p")
    figure.savefig(out / name, dpi=180)
    plt.close(figure)


def summarize(rows):
    failures = []
    for row in rows:
        reasons = []
        if row["error_prediction_rel_error"] > ERROR_REL_TOL:
            reasons.append("error magnitude discrepancy")
        if row["sensitivity_rel_error"] is None:
            if row["sensitivity_abs_error"] > NEAR_ZERO:
                reasons.append("near-zero slope absolute discrepancy")
        elif row["sensitivity_rel_error"] > SLOPE_REL_TOL:
            reasons.append("sensitivity discrepancy")
        if row["direction_agrees"] is False:
            reasons.append("opposite sensitivity sign")
        if reasons:
            failures.append({key: row[key] for key in (
                "p", "nf", "ar", "E", "E_hat", "E_res", "S_E", "S_E_hat",
                "error_prediction_rel_error", "sensitivity_abs_error",
                "sensitivity_rel_error", "q_max", "max_exact_fresnel_phase_defect_rad",
            )} | {"reasons": reasons})
    return {
        "samples": len(rows),
        "max_area_abs_relative_error": max(row["area_abs_relative_error"] for row in rows),
        "max_magnitude_prediction_rel_error": max(row["error_prediction_rel_error"] for row in rows),
        "max_sensitivity_rel_error": max((row["sensitivity_rel_error"] for row in rows
                                          if row["sensitivity_rel_error"] is not None), default=None),
        "max_sensitivity_abs_error": max(row["sensitivity_abs_error"] for row in rows),
        "fresnel_error_above_5_percent": sum(row["direct_error_exceeds_5_percent"] for row in rows),
        "quartic_phase_above_pi_over_8": sum(row["phase_indicator_exceeds_pi_over_8"] for row in rows),
        "prediction_failures": failures,
    }


def print_plan(selected, shapes, orders, nfs, ars, roi_order, quick):
    levels = orders[-1]
    count = len(shapes)
    calls = 0
    pairs = 0
    if "sensitivity" in selected:
        for p in shapes:
            grid = (*orders, 160) if p == 1.5 and not quick else orders
            calls += 3*len(grid) + 9
            pairs += 12*(roi_order**2*(sum(n*n for n in grid)+2*grid[-1]**2)
                         + (roi_order+4)**2*grid[-1]**2)
    for name, conditions in (("nf", nfs), ("ar", ars)):
        if name in selected:
            calls += 3*count*len(conditions)
            pairs += 12*count*len(conditions)*levels**2*roi_order**2
            calls += 2*12
            pairs += 24*(roi_order**2*(orders[-2]**2+2*levels**2)
                         + (roi_order+4)**2*levels**2)
    max_order = 160 if "sensitivity" in selected and not quick else levels
    memory = BATCH*4*max_order**2*128 / 2**20
    print(f"Studies: {', '.join(selected)}; p={shapes}; aperture orders={orders}; "
          f"ROI order={roi_order}; h={STEPS}")
    print(f"NF={nfs}; AR={ars}; upper estimate {calls} field evaluations, "
          f"{pairs:.3e} aperture/observation point pairs.")
    print(f"Peak quadrature workspace budget ~ {memory:.1f} MiB "
          "(array estimate, excluding Python, audit, and plotting overhead).")
    return {"estimated_field_evaluations_upper": calls, "estimated_point_pairs_upper": pairs,
            "estimated_batch_workspace_mib": memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "quick", "sensitivity", "nf-sweep", "ar-sweep", "all"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    explicit = {"sensitivity": args.sensitivity, "nf": args.nf_sweep, "ar": args.ar_sweep}
    selected = [key for key in explicit if args.all or not any(explicit.values()) or explicit[key]]
    shapes = (1., 1.5, 2., 4., 12.) if args.quick else SHAPES
    nfs = (0.1, 1., 10.) if args.quick else FRESNEL_NUMBERS
    ars = (1., 4.) if args.quick else ASPECT_RATIOS
    orders = (16, 24, 32) if args.quick else ORDERS
    roi_order = 8 if args.quick else 16
    cost = print_plan(selected, shapes, orders, nfs, ars, roi_order, args.quick)
    if args.plan or not (args.quick or args.all or any(explicit.values())):
        print("Plan only. No integrations performed.")
        return
    audit = independent_checks()
    evaluate = evaluator()
    report = {
        "quick": args.quick, "plan": cost, "audit": audit, "sweeps": {},
        "study_thresholds": {
            "magnitude_prediction_relative": ERROR_REL_TOL, "sensitivity_relative": SLOPE_REL_TOL,
            "near_zero_slope": NEAR_ZERO, "quartic_phase_indicator_rad": float(PHASE_LIMIT),
            "direct_fresnel_error_indicator": 0.05,
        },
        "notes": [
            "E, E_hat and E_res all use the direct RS-I denominator; E_hat_legacy "
            "is the RS-independent scalar prediction from the earlier normalization.",
            "The circled samples in error plots exceed the quartic-phase or direct Fresnel-error indicator.",
            "Phase alignment uses weighted complex overlap and rotates phase only.",
            "A large omitted phase or q can explain Taylor limitations, but the phase "
            "indicator is a diagnostic and does not prove a cause for each discrepancy.",
            "Convergence CSV varies aperture order, ROI order and h independently at the baseline "
            "and checks p=1.5 and p=12 at the largest selected NF and AR.",
            "At p=1 and p=12 second-order one-sided derivatives respect the study interval.",
        ],
    }
    out = args.out or Path("moment_sensitivity_quick" if args.quick else "moment_sensitivity_results")
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    if "sensitivity" in selected:
        print("Computing baseline and convergence...", flush=True)
        rows, baseline_checks = convergence(evaluate, shapes, orders, roi_order, Condition(),
                                            extra_order=not args.quick)
        checks.extend(baseline_checks)
        save_csv(out / "moment_sensitivity.csv", rows)
        plots(out, "moment_prediction.png", {"": rows}, ("error",))
        plots(out, "shape_sensitivity.png", {"": rows}, ("sensitivity",))
        report["sweeps"]["baseline"] = summarize(rows)
        report["convergence_max_change"] = {
            factor: {key: max(r[f"{key}_abs_change_vs_reference"] for r in baseline_checks
                             if r["varied"] == factor)
                     for key in VALUES}
            for factor in ("aperture_order", "finite_difference_step", "roi_order")
        }
    for name, values, condition, stem in (
        ("nf", nfs, lambda v: Condition(nf=v), "fresnel_number"),
        ("ar", ars, lambda v: Condition(ar=v), "aspect_ratio"),
    ):
        if name not in selected:
            continue
        groups = {}
        for value in values:
            print(f"Computing {name.upper()}={value:g}...", flush=True)
            groups[f"{name.upper()}={value:g}"] = [
                sensitivity(evaluate, p, condition(value), orders[-1], roi_order, STEPS[-1])
                for p in shapes
            ]
        save_csv(out / f"{stem}_sweep.csv", [row for group in groups.values() for row in group])
        plots(out, f"{stem}_comparison.png", groups, ("error", "sensitivity"))
        report["sweeps"][name] = {label: summarize(rows) for label, rows in groups.items()}
        _, extreme_checks = convergence(evaluate, (1.5, 12.), orders[-2:], roi_order,
                                        condition(values[-1]))
        checks.extend(extreme_checks)
    save_csv(out / "sensitivity_convergence.csv", checks)
    (out / "moment_validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
