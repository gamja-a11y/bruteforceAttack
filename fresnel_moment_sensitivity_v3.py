"""Check shape-sensitivity convergence against the archived v2 NF/theta sweep."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from fresnel_moment_sensitivity import AREA, BATCH, NEAR_ZERO, RADIUS, half_axes, save_csv
from fresnel_moment_sensitivity_v2 import Condition, context, evaluator, verify_conditions


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "moment_sensitivity_v2_results" / "nf_theta_2d_sweep.csv"
NFS = (0.1, 1., 10.)
THETAS = (0.02, 0.1, 0.3)
SHAPES = (1., 1.5, 2., 8., 12.)
APERTURE_ORDERS = (64, 96, 128, 160)
ROI_ORDERS = (20, 24, 32)
STEPS = (0.02, 0.01, 0.005)
FIELDS = ("E", "E_hat", "E_res")
SLOPES = ("S_E", "S_E_hat")
METRICS = FIELDS + SLOPES
REL_TOL = 0.01
ABS_TOL = 1e-7
AREA_TOL = 1e-6


def load_v2(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    archive = {}
    for row in rows:
        key = tuple(float(row[k]) for k in ("nf", "theta_ap", "p"))
        if key in archive:
            raise ValueError(f"Duplicate v2 case {key}")
        archive[key] = row
    if len(archive) != 200:
        raise ValueError(f"Expected the 200-case v2 sweep, found {len(archive)}")
    return archive


def stencil(p, h):
    if p == SHAPES[0]:
        return "forward_2", (p, round(p+h, 10), round(p+2*h, 10)), (-3, 4, -1)
    if p == SHAPES[-1]:
        return "backward_2", (p, round(p-h, 10), round(p-2*h, 10)), (3, -4, 1)
    return "centered_2", (p, round(p-h, 10), round(p+h, 10)), (0, -1, 1)


def verify_stencils():
    for p in SHAPES:
        for h in STEPS:
            _, points, coefficients = stencil(p, h)
            assert all(SHAPES[0] <= point <= SHAPES[-1] for point in points)
            np.testing.assert_allclose(np.dot(coefficients, np.square(points))/(2*h),
                                       2*p, rtol=1e-10, atol=1e-10)


def settings(aperture, roi, steps):
    a, r, h = aperture[-1], roi[-1], steps[-1]
    return {
        "aperture_order": [(order, r, h) for order in aperture],
        "roi_order": [(a, order, h) for order in roi],
        "fd_step": [(a, r, step) for step in steps],
    }


def archive_settings(row):
    return int(row["aperture_order"]), int(row["roi_order"]), float(row["fd_h"])


def plan(cases, axes, archive):
    count = pairs = max_order = 0
    for nf, theta, p in cases:
        configs = [cfg for sequence in axes.values() for cfg in sequence]
        configs.append(archive_settings(archive[nf, theta, p]))
        evaluations = {(sample, a, r) for a, r, h in configs for sample in stencil(p, h)[1]}
        count += len(evaluations)
        pairs += sum(4*a*a*r*r for _, a, r in evaluations)
        max_order = max(max_order, max(a for _, a, _ in evaluations))
    result = {"physical_cases": len(cases), "field_evaluations_upper": count,
              "aperture_observation_point_pairs_upper": pairs,
              "batch_array_budget_mib": BATCH*4*max_order**2*128/2**20}
    print(f"{len(cases)} physical cases; {count:,} field evaluations at most; "
          f"{pairs:,} aperture/observation point pairs at most.")
    print(f"Batch array budget ~ {result['batch_array_budget_mib']:.1f} MiB "
          "(excludes interpreter and plots).")
    print("Each refinement axis fixes the other two at their finest settings.", flush=True)
    return result


def evaluate_setting(evaluate, p, condition, config):
    order, roi, h = config
    name, points, coefficients = stencil(p, h)
    samples = [evaluate(q, condition, order, roi) for q in points]
    ax, ay = half_axes(p, condition.ar)
    nf_check = RADIUS**2/(condition.wavelength*condition.z)
    theta_check = RADIUS/condition.z
    np.testing.assert_allclose([nf_check, theta_check], [condition.nf, condition.theta],
                               rtol=2e-14, atol=0)
    # Subtract the center first to avoid cancellation in very small derivatives.
    slopes = [
        sum(weight*(sample[key]-samples[0][key]) for weight, sample in zip(coefficients, samples))/(2*h)
        for key in FIELDS[:2]
    ]
    abs_error = abs(slopes[0]-slopes[1])
    area_error = max(sample["area_relative_error"] for sample in samples)
    return {
        **context(p, condition, order, roi), **samples[0],
        "a_x": ax, "a_y": ay, "area_target": float(AREA),
        "stencil_max_area_relative_error": area_error, "area_check_pass": area_error <= AREA_TOL,
        "nf_reconstructed": nf_check, "theta_reconstructed": theta_check,
        "fd_h": h, "stencil": name, "S_E": slopes[0], "S_E_hat": slopes[1],
        "sensitivity_abs_error": abs_error,
        "sensitivity_rel_error": None if abs(slopes[0]) < NEAR_ZERO else abs_error/abs(slopes[0]),
    }


def delta_metrics(value, reference, label):
    result = {}
    for key in METRICS:
        delta = abs(value[key]-reference[key])
        denominator = abs(reference[key])
        minimum = NEAR_ZERO if key in SLOPES else 0
        result[f"{key}_abs_change_vs_{label}"] = delta
        result[f"{key}_rel_change_vs_{label}"] = delta/denominator if denominator > minimum else None
    return result


def tolerance_ratio(test, reference, key):
    return abs(test[key]-reference[key]) / max(ABS_TOL, REL_TOL*abs(reference[key]))


def verify_archive_conditions(row, condition):
    expected = {"wavelength": condition.wavelength, "z": condition.z, "ar": condition.ar,
                "radius": RADIUS, "area": AREA, "nf": condition.nf, "theta_ap": condition.theta}
    for key, value in expected.items():
        np.testing.assert_allclose(float(row[key]), value, rtol=2e-14, atol=0)


def compare_case(evaluate, nf, theta, p, axes, archived_row, quick):
    condition = Condition(nf, theta)
    verify_archive_conditions(archived_row, condition)
    finest_setting = axes["aperture_order"][-1]
    finest = evaluate_setting(evaluate, p, condition, finest_setting)
    old = {key: float(archived_row[key]) for key in METRICS}
    reproduced = evaluate_setting(evaluate, p, condition, archive_settings(archived_row))
    for key in METRICS:
        np.testing.assert_allclose(reproduced[key], old[key], rtol=2e-7, atol=5e-12,
                                   err_msg=f"v2 replay mismatch at NF={nf}, theta={theta}, p={p}, {key}")

    rows = []
    ratios = {}
    for axis, configs in axes.items():
        sequence = [evaluate_setting(evaluate, p, condition, cfg) for cfg in configs]
        ratios[axis] = {key: tolerance_ratio(sequence[-2], finest, key) for key in METRICS}
        level_key = {"aperture_order": "aperture_order", "roi_order": "roi_order", "fd_step": "fd_h"}[axis]
        for index, value in enumerate(sequence):
            row = {"varied": axis, "level": value[level_key], **value}
            row.update(delta_metrics(value, finest, "finest"))
            if index:
                row.update(delta_metrics(value, sequence[index-1], "previous"))
            rows.append(row)

    slope_max = max(ratios[axis][key] for axis in ratios for key in SLOPES)
    field_max = max(ratios[axis][key] for axis in ratios for key in FIELDS)
    v2_max = max(tolerance_ratio(old, finest, key) for key in SLOPES)
    area_pass = all(row["area_check_pass"] for row in (*rows, reproduced))
    status = {
        "convergence_status": ("quick_only" if quick else "within_last_step_tolerance"
                               if slope_max <= 1 and field_max <= 1 and area_pass
                               else "outside_last_step_tolerance"),
        "sensitivity_last_step_exceeds_tolerance": slope_max > 1,
        "field_last_step_exceeds_tolerance": field_max > 1,
        "max_sensitivity_tolerance_ratio": slope_max, "max_field_tolerance_ratio": field_max,
        "v2_sensitivity_exceeds_tolerance": v2_max > 1, "v2_max_sensitivity_tolerance_ratio": v2_max,
        "all_stencil_areas_pass": area_pass,
    }
    rows.append({"varied": "v2_replay", "level": None, **reproduced,
                 **delta_metrics(reproduced, finest, "finest")})
    archive_columns = {f"v2_{key}": value for key, value in old.items()}
    rows = [{**row, **archive_columns, **delta_metrics(row, old, "v2"), **status}
            for row in rows]
    summary = {
        "nf": nf, "theta_ap": theta, "p": p, **status, "ratios_by_axis": ratios,
        "finest": {key: finest[key] for key in (*METRICS, "aperture_order", "roi_order", "fd_h")},
        "v2": old, "finest_delta_from_v2": delta_metrics(finest, old, "v2"),
    }
    return rows, summary


def heatmap(path, summaries, quick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    columns = sorted({(row["nf"], row["theta_ap"]) for row in summaries})
    ps = sorted({row["p"] for row in summaries})
    grid = {(r["nf"], r["theta_ap"], r["p"]): r for r in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    panels = (("max_sensitivity_tolerance_ratio", "Last refinement vs tolerance"),
              ("v2_max_sensitivity_tolerance_ratio", "v2 vs finest tested setting"))
    maximum = max(10., max(row[key] for row in summaries for key, _ in panels))
    for ax, (key, title) in zip(axes, panels):
        matrix = np.full((len(ps), len(columns)), np.nan)
        for i, p in enumerate(ps):
            for j, pair in enumerate(columns):
                row = grid.get((*pair, p))
                if row is not None:
                    matrix[i, j] = max(0.01, row[key])
                    if row[key] > 1:
                        ax.plot(j, i, "x", color="white", markersize=7)
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis",
                          norm=LogNorm(vmin=0.01, vmax=maximum))
        fig.colorbar(image, ax=ax, label="Change / tolerance", shrink=.8)
        ax.set(title=title, xlabel="(NF, aperture parameter)",
               xticks=range(len(columns)), xticklabels=[f"{n:g}, {t:g}" for n, t in columns],
               yticks=range(len(ps)), yticklabels=[f"{p:g}" for p in ps])
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Superellipse exponent p")
    title = "Quick check" if quick else "Shape sensitivity convergence"
    fig.suptitle(f"{title} · × exceeds the numerical tolerance")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a reduced verification, not a convergence claim")
    parser.add_argument("--case", type=float, nargs=3, metavar=("NF", "THETA", "P"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    archive = load_v2(args.archive)
    full = list(product(NFS, THETAS, SHAPES))
    if args.case and tuple(args.case) not in full:
        parser.error("--case must be one of the requested NF, theta and p combinations")
    quick_cases = [(1., 0.1, p) for p in (1., 1.5, 2., 12.)] + [(10., 0.3, 12.)]
    cases = [tuple(args.case)] if args.case else quick_cases if args.quick else full
    ap = (32, 48) if args.quick else APERTURE_ORDERS
    roi = (12, 16) if args.quick else ROI_ORDERS
    axes = settings(ap, roi, STEPS)
    missing = [case for case in cases if case not in archive]
    if missing:
        parser.error(f"v2 archive is missing matching cases: {missing}")
    print(f"Aperture orders {ap}; ROI orders {roi}; steps {STEPS}.")
    cost = plan(cases, axes, archive)
    if args.plan or not (args.run or args.quick):
        print("Plan only. No field integrations performed.")
        return
    verify_stencils()
    verify_conditions(NFS, THETAS)
    default = ("moment_sensitivity_v3_quick" if args.quick else
               "moment_sensitivity_v3_case" if args.case else "moment_sensitivity_v3_results")
    out = (args.out or ROOT / default).resolve()
    if out == args.archive.resolve().parent or out == ARCHIVE.parent.resolve():
        parser.error("Choose a new output directory to preserve v2 results")
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    evaluate = evaluator()
    for index, (nf, theta, p) in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] NF={nf:g}, theta={theta:g}, p={p:g}", flush=True)
        more, summary = compare_case(evaluate, nf, theta, p, axes, archive[nf, theta, p], args.quick)
        rows.extend(more)
        summaries.append(summary)
    columns = dict.fromkeys(key for row in rows for key in row)
    save_csv(out / "sensitivity_convergence_v3.csv",
             [{key: row.get(key) for key in columns} for row in rows])
    heatmap(out / "sensitivity_convergence_heatmap.png", summaries, args.quick)
    report = {
        "plan": cost, "quick": args.quick, "archived_v2_rows": len(archive),
        "archived_v2_path": str(args.archive.resolve()),
        "v2_cases_not_checked": len(archive)-len(summaries), "meep_executed": False,
        "thresholds": {"relative": REL_TOL, "absolute_floor": ABS_TOL,
                       "near_zero_slope": NEAR_ZERO, "area_relative": AREA_TOL},
        "notes": [
            "Each axis is varied separately with the other two held at their finest values.",
            "Ratios use abs(change)/max(absolute_floor, relative*abs(finest)); >1 exceeds tolerance.",
            "The last pair on each axis is a finite-resolution stability test, not an error bound.",
            "A v2 discrepancy can occur even when the v3 last steps pass, and vice versa.",
            "Near-zero sensitivities are compared in absolute terms, without a relative percentage.",
            "All trial stencils are checked for area conservation by quadrature. NF and theta are "
            "reconstructed from wavelength, distance and fixed effective radius.",
        ],
        "cases": summaries,
        "outside_last_step_tolerance": [r for r in summaries
                                        if r["convergence_status"] == "outside_last_step_tolerance"],
        "v2_sensitivity_discrepancies": [r for r in summaries if r["v2_sensitivity_exceeds_tolerance"]],
    }
    (out / "convergence_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
