"""Numerical quadrature audit for a fixed superellipse and propagation geometry."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import beta, roots_legendre

from fresnel_moment_sensitivity import ROI as V3_ROI_RADIUS, observation_nodes


NF, THETA, ASPECT_RATIO, P = 10., 0.3, 1., 12.
RADIUS, ROI_RADIUS = 10., 5.
AREA = np.pi * RADIUS**2
Z = RADIUS / THETA
WAVELENGTH = RADIUS * THETA / NF
K = 2 * np.pi * Z / WAVELENGTH
PREF = 1 / (1j * WAVELENGTH * Z)
HALF_AXIS = np.sqrt(AREA / (4 * beta(1/P, 1+1/P)/P))
Q_MAX = ((2**(0.5-1/P)*HALF_AXIS+ROI_RADIUS)/Z)**2
FIELDS = ("U_RS", "U_F", "M0", "M2", "M4")
NORMS = ("E", "E_hat", "E_res")
APERTURE_ORDERS = (64, 96, 128, 160, 192, 256)
ROI_ORDERS = (16, 20, 24, 32, 40, 48, 64)
APERTURE_COMPARISON_ROI = 32
BATCH = 2
NEAR_ZERO = 1e-8
EXCELLENT, CONVERGED = 1e-4, 1e-3
ABS_EXCELLENT, ABS_CONVERGED = 1e-10, 1e-9
ADAPTIVE_TOL = 1e-9
PROBES = ((0., 0.), (2.5, 0.), (2., 2.), (3., 4.))
SCALES = (AREA*abs(PREF), AREA*abs(PREF), AREA, AREA*Q_MAX, AREA*Q_MAX**2)


@lru_cache(maxsize=None)
def unit_legendre(order):
    nodes, weights = roots_legendre(order)
    return (nodes+1)/2, weights/2


def aperture_rule(order):
    """Use Cartesian ray slopes in eight sectors, integrating to the exact edge."""
    r, wr = unit_legendre(order)
    t, wt = unit_legendre(order)
    edge = HALF_AXIS * (1+t**P)**(-1/P)
    # First octant: x=r*edge(t), y=t*x. The determinant is r*edge(t)**2.
    x = np.outer(r, edge).ravel()
    y = np.outer(r, t*edge).ravel()
    sector_weights = np.outer(wr*r, wt*edge**2).ravel()
    xs = np.concatenate((x, -x, -x, x, y, -y, -y, y))
    ys = np.concatenate((y, y, -y, -y, x, x, -x, -x))
    weights = np.tile(sector_weights, 8)
    analytic_x2 = 4*HALF_AXIS**4*beta(3/P, 1+1/P)/P
    integral_x2, integral_y2 = (float(np.dot(weights, axis**2)) for axis in (xs, ys))
    mapped_max = float(np.max((abs(xs)/HALF_AXIS)**P+(abs(ys)/HALF_AXIS)**P))
    audit = {
        "order": order, "nodes": weights.size, "area_numeric": float(weights.sum()),
        "area_relative_error": float(abs(weights.sum()/AREA-1)),
        "x2_integral_relative_error": float(abs(integral_x2/analytic_x2-1)),
        "y2_integral_relative_error": float(abs(integral_y2/analytic_x2-1)),
        "max_normalized_boundary_expression": mapped_max,
        "min_weight": float(weights.min()),
    }
    audit["passed"] = bool(mapped_max <= 1+1e-13 and weights.min() > 0 and max(
        audit[key] for key in ("area_relative_error", "x2_integral_relative_error",
                               "y2_integral_relative_error")) <= 1e-10)
    return xs, ys, weights, audit


def roi_rule(order):
    # v3 reaches this same rule through the v2 evaluator. Share its exact
    # node ordering and normalization while keeping aperture integration independent.
    if ROI_RADIUS != V3_ROI_RADIUS:
        raise RuntimeError("This validation ROI radius differs from v3")
    x, y, measure = observation_nodes(order)
    if not np.isclose(measure.sum(), 1., atol=1e-13, rtol=0) or np.any(measure <= 0):
        raise RuntimeError(f"Invalid normalized disk quadrature weights at order {order}")
    return x, y, measure


def gauss_fields(rule, x, y):
    xa, ya, weights, _ = rule
    out = np.empty((len(FIELDS), x.size), dtype=np.complex128)
    for start in range(0, x.size, BATCH):
        part = slice(start, start+BATCH)
        q = ((x[part, None]-xa)**2+(y[part, None]-ya)**2)/Z**2
        root = np.sqrt(1+q)
        rs_kernel = np.exp(1j*K*q/(root+1))*(1+1j/(K*root))/(1+q)
        rs = np.sum(weights*rs_kernel, axis=1)*PREF
        phase = K*q/2
        # Compute the Fresnel integral directly with separate real quadratures.
        fr = (np.cos(phase) @ weights + 1j*(np.sin(phase) @ weights))*PREF
        base = np.exp(1j*phase)*weights
        moment2 = q*base
        out[:, part] = (rs, fr, np.sum(base, axis=1), np.sum(moment2, axis=1),
                        np.sum(q*moment2, axis=1))
    return out


def adaptive_point(x, y):
    """Integrate Cartesian slices with adaptive Gauss-Kronrod, independently of the sectors."""
    # Scaling prevents the largest vector component from setting all tolerances.
    integral_scales = np.array([AREA, AREA, AREA, AREA*Q_MAX, AREA*Q_MAX**2, AREA])
    max_inner_error = 0.
    inner_success = True
    inner_evaluations = 0

    def sliced(u):
        nonlocal max_inner_error, inner_success, inner_evaluations
        width = HALF_AXIS*max(0., 1-abs(u)**P)**(1/P)

        def integrand(v):
            q = ((x-HALF_AXIS*u)**2+(y-v)**2)/Z**2
            root = np.sqrt(1+q)
            rs = np.exp(1j*K*(root-1))*(1+1j/(K*root))/(1+q)
            phase = K*q/2
            base = np.exp(1j*phase)
            return HALF_AXIS*np.array([rs, np.cos(phase)+1j*np.sin(phase), base,
                                       q*base, q*q*base, 1])/integral_scales

        val, err, info = quad_vec(integrand, -width, width, epsabs=ADAPTIVE_TOL/8,
                                  epsrel=ADAPTIVE_TOL/8, norm="max", limit=200,
                                  full_output=True)
        max_inner_error = max(max_inner_error, float(err))
        inner_success &= bool(info.success)
        inner_evaluations += info.neval
        return val

    value, err, info = quad_vec(sliced, -1., 1., points=[-.99, -.9, 0., .9, .99],
                                epsabs=ADAPTIVE_TOL, epsrel=ADAPTIVE_TOL,
                                norm="max", limit=400, full_output=True)
    value *= integral_scales
    value[:2] *= PREF
    return value[:5], {
        "adaptive_ok": bool(info.success and inner_success),
        "adaptive_outer_scaled_error_estimate": float(err),
        "adaptive_inner_max_scaled_error_estimate": max_inner_error,
        "adaptive_outer_evaluations": info.neval,
        "adaptive_inner_evaluations": inner_evaluations,
        "adaptive_area_relative_error": float(abs(value[5]/AREA-1)),
    }


def weighted_norm(value, weights):
    return float(np.sqrt(np.dot(weights, abs(value)**2)))


def ratio_or_none(value, denominator):
    return float(value/denominator) if denominator else None


def field_delta(current, previous, weights, scale):
    delta = abs(current-previous)
    reference = weighted_norm(previous, weights)
    abs_l2 = weighted_norm(delta, weights)
    phase_weights = weights*abs(previous)**2
    phase = np.angle(current*previous.conj())
    weak = abs(previous) <= NEAR_ZERO*scale
    return {
        "abs_complex_l2": abs_l2, "relative_complex_l2": ratio_or_none(abs_l2, reference),
        "reference_l2": reference, "max_abs_complex_difference": float(delta.max()),
        "intensity_l1_relative": ratio_or_none(np.dot(weights, abs(abs(current)**2-abs(previous)**2)),
                                               phase_weights.sum()),
        "weighted_phase_rms_rad": ratio_or_none(weighted_norm(phase, phase_weights), reference),
        "near_zero_point_count": int(weak.sum()),
        "near_zero_max_abs_difference": float(delta[weak].max()) if np.any(weak) else None,
    }


def classify(relative, absolute, reference_amplitude, scale, weak_max=None):
    absolute_mode = reference_amplitude <= NEAR_ZERO*scale
    score = absolute/scale if absolute_mode else relative
    excellent = ABS_EXCELLENT if absolute_mode else EXCELLENT
    converged = ABS_CONVERGED if absolute_mode else CONVERGED
    weak_scaled = 0. if weak_max is None else weak_max/scale
    if score is None or score > converged or weak_scaled > ABS_CONVERGED:
        return "needs_more_validation"
    if score <= excellent and weak_scaled <= ABS_EXCELLENT:
        return "excellent"
    return "converged"


def fresnel_identity(fields, weights):
    absolute = weighted_norm(fields[1]-PREF*fields[2], weights)
    reference = weighted_norm(fields[1], weights)
    return {
        "UF_vs_M0_absolute_l2": absolute,
        "UF_vs_M0_relative_l2": ratio_or_none(absolute, reference),
        "UF_vs_M0_pass": bool(absolute <= 1e-12*SCALES[1]),
    }


def roi_metrics(fields, weights):
    rs, fr, m0, m2, m4 = fields
    c0 = 1+1j/K
    c1 = -1-1.5j/K
    c2 = 1+15j/(8*K)-1j*K*c0/8
    correction = PREF*((c0-1)*m0+c1*m2+c2*m4)
    reference = weighted_norm(rs, weights)
    if reference == 0:
        raise RuntimeError("RS-I field has zero norm on the ROI")
    return {
        "E": weighted_norm(rs-fr, weights)/reference,
        "E_hat": weighted_norm(correction, weights)/reference,
        "E_res": weighted_norm(rs-fr-correction, weights)/reference,
        "rs_reference_norm": reference,
    }


def add_metric_changes(row, previous):
    if previous is None:
        return
    for name in NORMS:
        change = abs(row[name]-previous[name])
        relative = ratio_or_none(change, abs(previous[name]))
        row[f"{name}_abs_change"] = change
        row[f"{name}_relative_change"] = relative
        row[f"{name}_step_status"] = classify(relative, change, abs(previous[name]), 1.)


def aperture_study(out, orders):
    x, y, weights = roi_rule(APERTURE_COMPARISON_ROI)
    rows, errors, audits, identities = [], [], [], []
    previous = None
    for order in orders:
        print(f"Aperture order {order} at fixed ROI order {APERTURE_COMPARISON_ROI}", flush=True)
        rule = aperture_rule(order)
        current = gauss_fields(rule, x, y)
        np.savez_compressed(out/f"aperture_order_{order}_fields.npz", x=x, y=y, weights=weights,
                            **dict(zip(FIELDS, current)))
        audits.append(rule[3])
        identities.append({"aperture_order": order, **fresnel_identity(current, weights)})
        error_row = {"aperture_order": order, "roi_order": APERTURE_COMPARISON_ROI,
                     "previous_aperture_order": errors[-1]["aperture_order"] if errors else None,
                     **roi_metrics(current, weights)}
        add_metric_changes(error_row, errors[-1] if errors else None)
        errors.append(error_row)
        for index, (name, field, scale) in enumerate(zip(FIELDS, current, SCALES)):
            row = {"field": name, "aperture_order": order, "roi_order": APERTURE_COMPARISON_ROI,
                   "field_weighted_l2": weighted_norm(field, weights), "natural_scale": float(scale)}
            if previous is not None:
                delta = field_delta(field, previous[index], weights, scale)
                row.update(previous_aperture_order=previous_order, **delta)
                row["step_status"] = classify(delta["relative_complex_l2"], delta["abs_complex_l2"],
                                              delta["reference_l2"], scale,
                                              delta["near_zero_max_abs_difference"])
            rows.append(row)
        previous, previous_order = current, order
    return rows, errors, audits, identities, rule, current


def roi_study(rule, fixed_fields, orders):
    rows = []
    previous = None
    aperture_order = rule[3]["order"]
    for order in orders:
        print(f"ROI order {order} at fixed aperture order {aperture_order}", flush=True)
        x, y, weights = roi_rule(order)
        fields = (fixed_fields if order == APERTURE_COMPARISON_ROI and fixed_fields is not None
                  else gauss_fields(rule, x, y))
        metrics = roi_metrics(fields, weights)
        row = {"roi_order": order, "aperture_order": aperture_order,
               "roi_weight_sum": float(weights.sum()), **metrics, **fresnel_identity(fields, weights)}
        if previous is not None:
            row["previous_roi_order"] = previous["roi_order"]
        add_metric_changes(row, previous)
        rows.append(row)
        previous = row
    return rows


def independent_comparison(rule, probes):
    rows = []
    coordinates = np.array(probes)
    gauss = gauss_fields(rule, coordinates[:, 0], coordinates[:, 1])
    for index, (x, y) in enumerate(probes):
        print(f"Adaptive comparison at ({x:g}, {y:g})", flush=True)
        adaptive, audit = adaptive_point(x, y)
        identity_abs = float(abs(adaptive[1]-PREF*adaptive[2]))
        for name, fixed, estimate, scale in zip(FIELDS, gauss[:, index], adaptive, SCALES):
            delta = float(abs(fixed-estimate))
            relative = ratio_or_none(delta, abs(estimate))
            rows.append({
                "x": x, "y": y, "field": name,
                "gauss_real": float(fixed.real), "gauss_imag": float(fixed.imag),
                "adaptive_real": float(estimate.real), "adaptive_imag": float(estimate.imag),
                "abs_difference": delta, "relative_difference": relative,
                "adaptive_UF_vs_M0_abs_difference": identity_abs,
                "crosscheck_status": classify(relative, delta, abs(estimate), scale),
                **audit,
            })
    return rows


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(out, stem, fields, rows, orders):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
    for name in fields:
        series = [row for row in rows if row["field"] == name and "previous_aperture_order" in row]
        for ax, key in zip(axes, ("relative_complex_l2", "abs_complex_l2")):
            values = [row for row in series if row[key] is not None and row[key] > 0]
            ax.semilogy([row["aperture_order"] for row in values], [row[key] for row in values],
                        "o-", label=name)
    for threshold in (EXCELLENT, CONVERGED):
        axes[0].axhline(threshold, color="0.5", linestyle=":", linewidth=0.8)
    for ax, label in zip(axes, ("Relative complex L2", "Absolute complex L2")):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1].set(xlabel="Aperture quadrature order", xticks=orders[1:])
    fig.suptitle("Change from the previous aperture order")
    fig.savefig(out/f"{stem}_convergence.png", dpi=170)
    plt.close(fig)


def plan(aperture_orders, roi_orders, do_aperture, do_roi, probes):
    finest = aperture_orders[-1]
    a = (sum(8*order**2*APERTURE_COMPARISON_ROI**2 for order in aperture_orders)
         if do_aperture else 0)
    b = (8*finest**2*sum(order**2 for order in roi_orders
                          if not do_aperture or order != APERTURE_COMPARISON_ROI)
         if do_roi else 0)
    c = 8*finest**2*len(probes)
    memory = BATCH*8*finest**2*128/2**20
    print(f"NF={NF:g}, theta={THETA:g}, AR={ASPECT_RATIO:g}, p={P:g}, area={AREA:.8g}, "
          f"z={Z:.8g}, wavelength={WAVELENGTH:g}, ROI radius={ROI_RADIUS:g}")
    if do_aperture:
        print(f"Aperture {aperture_orders} at fixed ROI order {APERTURE_COMPARISON_ROI}: {a:,} point pairs")
    if do_roi:
        print(f"ROI {roi_orders} at fixed aperture order {finest}: {b:,} additional point pairs")
    print(f"Adaptive cross-check: {len(probes)} probes; {c:,} fixed-rule comparison pairs.")
    print(f"Total fixed-rule pairs: {a+b+c:,}; batch array budget ~ {memory:.0f} MiB. "
          "Adaptive evaluation count and additional runtime memory are not included.", flush=True)
    return {"aperture_pairs": a, "roi_additional_pairs": b, "probe_gauss_pairs": c,
            "total_fixed_pairs": a+b+c, "workspace_budget_mib": memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--aperture-only", action="store_true", help="Run only the aperture-order study")
    scope.add_argument("--roi-only", action="store_true", help="Run only the observation-order study")
    parser.add_argument("--quick", action="store_true", help="Use fewer orders and probes; requires --run to integrate")
    parser.add_argument("--out", type=Path, help="A new output directory; existing directories are preserved")
    args = parser.parse_args()
    do_aperture, do_roi = not args.roi_only, not args.aperture_only
    aperture_orders = (32, 48, 64) if args.quick else APERTURE_ORDERS
    roi_orders = (16, 24, 32) if args.quick else ROI_ORDERS
    probes = (PROBES[0], PROBES[-1]) if args.quick else PROBES
    if not (do_aperture and do_roi):
        probes = ()
    cost = plan(aperture_orders, roi_orders, do_aperture, do_roi, probes)
    if not args.run:
        print("Plan only. No quadrature, output files, or FDTD executed.")
        return
    label = "_aperture" if args.aperture_only else "_roi" if args.roi_only else ""
    label += "_quick" if args.quick else ""
    out = args.out or Path(__file__).resolve().parent / f"fresnel_quadrature_results{label}"
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output path already exists: {out}. Choose a new --out.")
    aperture, aperture_errors, identities = [], [], []
    if do_aperture:
        aperture, aperture_errors, mapping, identities, rule, fixed = aperture_study(out, aperture_orders)
        for stem, entries in (("rs", ("U_RS",)), ("fresnel", ("U_F",)),
                              ("moment", ("M0", "M2", "M4"))):
            write_csv(out/f"{stem}_convergence.csv",
                      [row for row in aperture if row["field"] in entries])
            plot(out, stem, entries, aperture, aperture_orders)
        write_csv(out/"aperture_error_metrics.csv", aperture_errors)
    else:
        rule = aperture_rule(aperture_orders[-1])
        mapping, fixed = [rule[3]], None
    roi = roi_study(rule, fixed, roi_orders) if do_roi else []
    if roi:
        write_csv(out/"roi_convergence.csv", roi)
    adaptive = independent_comparison(rule, probes) if probes else []
    adaptive_area_ok = (all(row["adaptive_area_relative_error"] <= 1e-9 for row in adaptive)
                        if adaptive else None)
    if adaptive:
        write_csv(out/"independent_quadrature_comparison.csv", adaptive)
    diagnoses = {}
    for name in FIELDS:
        if not do_aperture:
            diagnoses[name] = {"status": "not_assessed", "reasons": ["ROI-only run"]}
            continue
        last = next(row for row in reversed(aperture) if row["field"] == name)
        checked = [row for row in adaptive if row["field"] == name]
        reasons = []
        if last["step_status"] == "needs_more_validation":
            reasons.append("last aperture refinement is too large; inspect oscillatory kernel sampling and edge mapping")
        if any(not row["adaptive_ok"] for row in checked):
            reasons.append("adaptive integration status did not confirm its tolerance")
        if any(row["crosscheck_status"] == "needs_more_validation" for row in checked):
            reasons.append("representative points disagree between sector and Cartesian quadrature")
        if any(not audit["passed"] for audit in mapping):
            reasons.append("geometry/Jacobian invariants failed")
        if adaptive_area_ok is False:
            reasons.append("Cartesian adaptive aperture area does not match the analytic area")
        if name in ("U_F", "M0") and (
            any(not item["UF_vs_M0_pass"] for item in (*identities, *roi))
            or any(row["adaptive_UF_vs_M0_abs_difference"] > 1e-12*SCALES[1]
                   for row in checked)
        ):
            reasons.append("direct Fresnel and reconstructed M0 field differ")
        diagnoses[name] = {
            "status": "needs_more_validation" if reasons else last["step_status"],
            "last_aperture_step": last, "reasons": reasons,
            "adaptive_crosscheck_performed": bool(checked),
        }
    report = {
        "physical_conditions": {"nf": NF, "theta_ap": THETA, "aspect_ratio": ASPECT_RATIO,
                                "p": P, "a_eff": RADIUS, "area": float(AREA), "z": Z,
                                "wavelength": WAVELENGTH, "roi_radius": ROI_RADIUS},
        "plan": cost, "quick": args.quick, "field_assessments": diagnoses,
        "last_aperture_error_metrics_step": aperture_errors[-1] if aperture_errors else None,
        "last_roi_step": roi[-1] if roi else None,
        "roi_diagnosis": (
            "Final ROI changes exceed tolerance with the aperture order fixed; refine disk quadrature."
            if any(roi[-1][f"{name}_step_status"] == "needs_more_validation" for name in NORMS)
            else "Final ROI changes meet tolerance at the fixed aperture order."
        ) if roi else "Not assessed in aperture-only mode.",
        "mapping_checks": mapping, "fresnel_M0_identity": identities,
        "adaptive_probe_count": len(probes), "adaptive_aperture_area_check_pass": adaptive_area_ok,
        "thresholds": {"relative_excellent": EXCELLENT, "relative_converged": CONVERGED,
                       "near_zero_natural_scale_fraction": NEAR_ZERO,
                       "absolute_excellent_scale_fraction": ABS_EXCELLENT,
                       "absolute_converged_scale_fraction": ABS_CONVERGED},
        "notes": [
            "Each order supplies separate radial/sector or radial/angular Gauss-Legendre rules.",
            "ROI coordinates and normalized weights use the same quarter-disk rule as v3.",
            "Field comparisons are raw complex differences.",
            "A quick result applies only to its reduced order sequence.",
            "Adaptive probes use nested Cartesian integration; inner and outer estimates are not rigorous bounds.",
            "For near-zero norms use absolute change divided by a natural scale. Near-zero sample points "
            "also have a separate maximum absolute-change check.",
            "Diagnostics address numerical integration only. No FDTD or shape derivative was executed.",
        ],
    }
    (out/"convergence_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
