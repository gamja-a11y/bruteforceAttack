"""Check radial and angular convergence on the same normalized circular ROI."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from fresnel_quadrature_validation import (
    AREA, K, NF, NORMS, P, PREF, ROI_RADIUS, THETA, WAVELENGTH, Z,
    add_metric_changes, aperture_rule, gauss_fields, roi_metrics, unit_legendre,
    write_csv,
)


APERTURE_ORDER = 64
ORDERS = (32, 40, 48, 64, 80, 96)
FIXED_ORDER = 96
EXTENDED_ORDER = 128
INDEPENDENT_ORDERS = (64, 96, 128)
TOL = 1e-3


def polar_rule(radial_order, angular_order):
    radial, wr = unit_legendre(radial_order)
    angle, wa = unit_legendre(angular_order)
    # At AR=1, the fields are invariant under sign changes and x/y exchange.
    # One octant represents the full disk and its analytically normalized area.
    angle = (np.pi/4)*angle
    return (ROI_RADIUS*np.outer(radial, np.cos(angle)).ravel(),
            ROI_RADIUS*np.outer(radial, np.sin(angle)).ravel(),
            np.outer(2*radial*wr, wa).ravel())


def cartesian_rule(order):
    """Cover the first quadrant with stretched Cartesian strips.

    x/R=1-u^2 removes the square-root endpoint in the strip height.
    y=v*sqrt(R^2-x^2); the determinant is 2*R^2*u^2*sqrt(2-u^2).
    """
    u, w = unit_legendre(order)
    x = ROI_RADIUS*(1-u*u)
    heights = ROI_RADIUS*u*np.sqrt(2-u*u)
    weights_x = 8/np.pi*u*u*np.sqrt(2-u*u)*w
    return np.repeat(x, order), np.outer(heights, u).ravel(), np.outer(weights_x, w).ravel()


def verify_roi(x, y, weights):
    r2 = x*x+y*y
    if np.any(r2 > ROI_RADIUS**2*(1+1e-14)) or np.any(weights <= 0):
        raise RuntimeError("Invalid observation nodes or weights")
    np.testing.assert_allclose(np.dot(weights, r2), ROI_RADIUS**2/2, rtol=2e-13)
    np.testing.assert_allclose(weights.sum(), 1., rtol=2e-13)


def integrals(aperture, rule):
    x, y, weights = rule
    verify_roi(x, y, weights)
    metrics = roi_metrics(gauss_fields(aperture, x, y), weights)
    return {**metrics, "roi_weight_sum": float(weights.sum()), "observation_points": x.size}


def passed_step(row):
    return all(row.get(f"{key}_relative_change") is not None
               and row[f"{key}_relative_change"] <= TOL for key in NORMS)


def stable(rows):
    return len(rows) >= 3 and all(passed_step(row) for row in rows[-2:])


def relative_differences(metrics, reference):
    return {key: abs(metrics[key]-reference[key])/abs(reference[key])
            if reference[key] != 0 else None for key in NORMS}


def within_tolerance(values):
    return all(value is not None and value <= TOL for value in values.values())


def print_plan(fixed_radial, fixed_angular, extend):
    pairs = {(n, fixed_angular) for n in ORDERS} | {(fixed_radial, n) for n in ORDERS}
    pairs.add((ORDERS[-1], ORDERS[-1]))
    base_nodes = sum(nr*na for nr, na in pairs)
    source_nodes = 8*APERTURE_ORDER**2
    plan = {
        "aperture_order": APERTURE_ORDER, "source_nodes": source_nodes,
        "base_polar_observation_points": base_nodes,
        "base_polar_point_pairs": base_nodes*source_nodes,
        "cartesian_point_pairs_initial": INDEPENDENT_ORDERS[0]**2*source_nodes,
        "cartesian_point_pairs_max": sum(n*n for n in INDEPENDENT_ORDERS)*source_nodes,
        "extra_polar_point_pairs_if_both_axes_extend":
            (EXTENDED_ORDER*(fixed_radial+fixed_angular)+EXTENDED_ORDER**2)*source_nodes
            if extend else 0,
    }
    print(f"Aperture fixed at {APERTURE_ORDER}; radial and angular sweep orders {ORDERS}; "
          f"fixed radial={fixed_radial}, fixed angular={fixed_angular}.")
    print(f"Polar plan: {base_nodes:,} unique observation points, "
          f"{plan['base_polar_point_pairs']:,} aperture/observation pairs. "
          f"Initial Cartesian check: {plan['cartesian_point_pairs_initial']:,} pairs.")
    if extend:
        print("An axis that fails the two-step test can extend to 128.")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--extend", action="store_true",
                        help="Extend any unresolved polar axis to order 128")
    parser.add_argument("--fixed-radial", type=int, default=FIXED_ORDER)
    parser.add_argument("--fixed-angular", type=int, default=FIXED_ORDER)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.fixed_radial, args.fixed_angular) < 16:
        parser.error("Fixed ROI orders must be at least 16")
    cost = print_plan(args.fixed_radial, args.fixed_angular, args.extend)
    if not args.run:
        return
    out = args.out or Path(__file__).resolve().parent/"fresnel_roi_order64_results"
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output path already exists: {out}. Choose a new --out.")

    aperture = aperture_rule(APERTURE_ORDER)
    if not aperture[3]["passed"]:
        raise RuntimeError("The fixed aperture rule failed its geometric invariants")

    @lru_cache(maxsize=None)
    def polar(nr, na):
        print(f"Polar: radial={nr}, angular={na}", flush=True)
        return integrals(aperture, polar_rule(nr, na))

    series = {}
    for axis in ("radial", "angular"):
        rows = []

        def append(order):
            nr, na = ((order, args.fixed_angular) if axis == "radial"
                      else (args.fixed_radial, order))
            row = {"varied": axis, "order": order, "radial_order": nr, "angular_order": na,
                   "aperture_order": APERTURE_ORDER, **polar(nr, na)}
            if rows:
                row["previous_order"] = rows[-1]["order"]
                add_metric_changes(row, rows[-1])
                row["all_relative_changes_le_1e-3"] = passed_step(row)
            rows.append(row)

        for order in ORDERS:
            append(order)
        if args.extend and not stable(rows):
            append(EXTENDED_ORDER)
        series[axis] = rows
        write_csv(out/f"{axis}_roi_convergence.csv", rows)

    nr, na = series["radial"][-1]["order"], series["angular"][-1]["order"]
    reference = polar(nr, na)
    if nr != FIXED_ORDER or na != FIXED_ORDER:
        interactions = relative_differences(reference, polar(FIXED_ORDER, FIXED_ORDER))
    else:
        interactions = dict.fromkeys(NORMS, 0.)
    mixed_checks = {axis: relative_differences(reference, rows[-1])
                    for axis, rows in series.items()}

    independent = []
    for order in INDEPENDENT_ORDERS:
        print(f"Cartesian cross-check: order={order}", flush=True)
        row = {"cartesian_order": order, "reference_radial_order": nr, "reference_angular_order": na,
               **integrals(aperture, cartesian_rule(order))}
        add_metric_changes(row, independent[-1] if independent else None)
        comparison = relative_differences(row, reference)
        row.update({f"{key}_abs_difference_from_polar": abs(row[key]-reference[key]) for key in NORMS})
        row.update({f"{key}_relative_difference_from_polar": value for key, value in comparison.items()})
        row["within_tolerance_of_polar"] = within_tolerance(comparison)
        independent.append(row)
        if row["within_tolerance_of_polar"]:
            break
    write_csv(out/"cartesian_roi_comparison.csv", independent)
    np.testing.assert_allclose([ROI_RADIUS, NF, THETA], [5., 10., .3], rtol=1e-14)
    passed = {axis: stable(rows) for axis, rows in series.items()}
    report = {
        "status": ("converged" if all(passed.values()) and independent[-1]["within_tolerance_of_polar"]
                   and all(within_tolerance(v) for v in mixed_checks.values()) else "needs_more_validation"),
        "relative_tolerance": TOL, "required_consecutive_passes": 2, "aperture_audit": aperture[3],
        "conditions": {"nf": NF, "theta_ap": THETA, "ar": 1, "p": P, "z": Z, "wavelength": WAVELENGTH,
                       "aperture_area": AREA, "roi_radius": ROI_RADIUS},
        "plan": cost, "consecutive_step_pass": passed,
        "last_two_steps": {axis: rows[-2:] for axis, rows in series.items()},
        "reference": {"radial_order": nr, "angular_order": na, **reference},
        "relative_changes_from_96_by_96": interactions,
        "mixed_order_check_relative_changes": mixed_checks,
        "cartesian_crosscheck": independent,
        "notes": [
            "The polar rule integrates one octant by reflection and interchange symmetry; "
            "the Cartesian reference integrates one quadrant using square-root-smoothed strips.",
            "Both rules normalize by the analytic disk area and use the same RS-I norm in "
            "each row's E, E_hat and E_res. The aperture rule is fixed across all rows.",
            "Success requires two consecutive accepted changes on each axis and "
            "agreement with an independent Cartesian rule.",
        ],
    }
    (out/"roi_convergence_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(f"{report['status']}: saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
