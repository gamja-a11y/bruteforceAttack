"""Audit Case A DFT time and spatial resolution convergence on fixed physical planes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fresnel_shape_sensitivity import Experiment
from meep_boundary_validation import (
    EXP, KEYS, ROOT, THICKNESS, Setup, comparison, incident_metrics, load_archive,
)
from meep_shape_diagnostic_v2 import phase_align, scalar_fields


TIMES = (150, 200, 250, 300)
RESOLUTIONS = (6, 8, 10, 12)
COURANT = 0.5
L2_TOLERANCE = 0.01


@dataclass(frozen=True)
class ScalarPlane(Experiment):
    z: float = 100.0

    @property
    def distance(self):
        # Keep the aperture radius and wavelength fixed when evaluating a plane.
        return self.z


def verify_phase_alignment():
    reference = np.array([1 + 2j, 3 - 1j, -2j])
    weights = np.array([0.2, 0.3, 0.5])
    field = 2 * reference * np.exp(1.2j)
    np.testing.assert_allclose(phase_align(field, reference), 2 * reference, atol=1e-13)
    orthogonal = np.array([0j, 2j])
    np.testing.assert_array_equal(phase_align(orthogonal, np.array([1j, 0j])), orthogonal)
    results = comparison(field, reference, weights, "test")
    np.testing.assert_allclose(
        [results["test_relative_l2_aligned"], results["test_intensity_l1"],
         results["test_phase_rms_raw_rad"], results["test_phase_rms_aligned_rad"]],
        [1, 3, 1.2, 0], atol=1e-13,
    )


def make_plan(time_sweep, resolution_sweep, quick):
    times = TIMES[:2] if quick else TIMES
    resolutions = RESOLUTIONS[:2] if quick else RESOLUTIONS
    jobs = {}
    if time_sweep:
        jobs[6] = times
    if resolution_sweep:
        for resolution in resolutions:
            jobs.setdefault(resolution, times[-2:])
    return times, resolutions, jobs


def plan_cost(jobs):
    settings = []
    for resolution, times in sorted(jobs.items()):
        domain = Setup().domain(resolution)
        cells = domain["estimated_yee_cells"]
        steps = int(np.ceil(times[-1] * resolution / COURANT))
        settings.append({
            "resolution": resolution, "times": list(times), "cell": domain["cell"],
            "yee_cells": cells, "fdtd_runs": 2, "steps_per_run": steps,
            "estimated_cell_updates": 2 * cells * steps,
            # Budgetary estimate, not an allocator prediction. One run at a time.
            "estimated_memory_gib_range": [cells * b / 2**30 for b in (64, 128)],
        })
    return settings


def xyz_axes(metadata):
    return tuple(np.asarray(axis, dtype=float).ravel() for axis in metadata[:3])


def check_regular_axis(axis, resolution):
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise RuntimeError("Missing Meep grid axis")
    if not np.allclose(np.diff(axis), 1 / resolution, rtol=1e-7, atol=1e-10):
        raise RuntimeError("Meep grid spacing does not match resolution")


def dft_snapshot(sim, monitor, component, x, y, requested_z, center_z, resolution):
    xs, ys, zs = xyz_axes(sim.get_array_metadata(dft_cell=monitor))
    values = np.asarray(sim.get_dft_array(monitor, component, 0)).squeeze()
    if values.shape != (xs.size, ys.size) or zs.size != 1 or not np.isfinite(zs).all():
        raise RuntimeError(f"Expected a collapsed Meep XY monitor, got {values.shape}")
    check_regular_axis(xs, resolution)
    check_regular_axis(ys, resolution)
    actual_z = float(zs[0] + center_z)
    delta_z = actual_z - requested_z
    tolerance = 64 * np.finfo(float).eps * max(1, abs(center_z), abs(requested_z))
    if abs(delta_z) > tolerance:
        # A collapsed monitor is defined at its requested plane by interpolation
        # between centered-grid samples. Never accept a one-cell coordinate error.
        raise RuntimeError(
            f"Collapsed DFT plane z={actual_z:.16g}, requested {requested_z:.16g}, "
            f"delta={delta_z:+.3g}; metadata do not describe the requested plane"
        )
    sampled = RegularGridInterpolator((xs, ys), values, bounds_error=True)(
        np.column_stack((x, y))
    )
    if not np.all(np.isfinite(sampled)):
        raise RuntimeError("Nonfinite DFT field on the archived observation grid")
    audit = {
        "requested_lab_z": requested_z, "actual_lab_z": actual_z, "delta_z": delta_z,
        "native_meep_x": xs.tolist(), "native_meep_y": ys.tolist(),
        "native_meep_z": zs.tolist(), "xy_sampling": "complex linear interpolation",
    }
    if hasattr(sim, "get_array_slice_dimensions"):
        # The min/max z corners are the centered-grid support before collapse.
        support = sim.get_array_slice_dimensions(vol=monitor.where)
        support_z = [float(v.z + center_z) for v in (support.min_corner, support.max_corner)]
        if not support_z[0] - tolerance <= actual_z <= support_z[1] + tolerance:
            raise RuntimeError(f"DFT plane z={actual_z} is outside grid support {support_z}")
        audit["native_centered_z_support_lab"] = support_z
    return sampled, audit


def inspect_screen(mp, sim, center_z, resolution):
    # Restrict the material read to the slab and one or more air layers on each side.
    width = 2 * (EXP.radius + 0.4)
    center = mp.Vector3(0, 0, -center_z)
    size = mp.Vector3(width, width, THICKNESS + 4 / resolution)
    xs, ys, zs = xyz_axes(sim.get_array_metadata(center=center, size=size))
    material = np.asarray(sim.get_array(center=center, size=size, component=mp.Dielectric))
    if material.shape != (xs.size, ys.size, zs.size):
        raise RuntimeError(f"Material field shape {material.shape} does not match its coordinates")
    for axis in (xs, ys, zs):
        check_regular_axis(axis, resolution)
    zz = zs + center_z
    radius = np.hypot(xs[:, None], ys[None, :])
    air_like = np.isclose(material, 1, rtol=0, atol=1e-6)
    # Use off-hole samples that are within the centered monitor, away from
    # the circular interface. PEC may be reported as nonfinite permittivity.
    off_hole = radius > EXP.radius + 1 / resolution
    material_fraction = np.mean(~air_like[off_hole], axis=0)
    detected = material_fraction >= 0.5
    summary = {
        "component": "Dielectric on centered grid", "z_lab": zz.tolist(),
        "off_hole_non_air_fraction_by_z": material_fraction.tolist(),
        "detected_slab_layer_count": int(detected.sum()),
        "nominal_thickness": THICKNESS, "nominal_radius": EXP.radius,
    }
    if np.any(detected):
        middle = np.flatnonzero(detected)[np.argmin(abs(zz[detected]))]
        aperture = air_like[:, :, middle]
        area = int(aperture.sum()) / resolution**2
        summary.update(
            raster_status="detected", effective_thickness_from_layers=float(detected.sum() / resolution),
            detected_layer_centers_lab=zz[detected].tolist(),
            measured_open_area=area, equivalent_open_radius=float(np.sqrt(area / np.pi)),
            boundary_disagreement_area=float(np.count_nonzero(aperture != (radius <= EXP.radius))
                                             / resolution**2),
            boundary_mask_z_lab=float(zz[middle]),
        )
    else:
        summary["raster_status"] = "PEC could not be identified from centered Dielectric samples"
    return summary, {"geometry_x": xs, "geometry_y": ys, "geometry_z_lab": zz,
                     "geometry_epsilon": material}


def simulate(mp, resolution, times, archive, with_screen):
    setup = Setup()
    domain = setup.domain(resolution)
    sx, sy, sz = domain["cell"]
    center_z = domain["center_z"]

    def position(z):
        return mp.Vector3(0, 0, z - center_z)

    geometry = []
    if with_screen:
        geometry = [
            mp.Block(size=mp.Vector3(sx, sy, THICKNESS), center=position(0), material=mp.metal),
            mp.Cylinder(radius=EXP.radius, height=THICKNESS + 2 / RESOLUTIONS[0],
                        axis=mp.Vector3(0, 0, 1), center=position(0), material=mp.air),
        ]
    frequency = 1 / EXP.wavelength
    source = mp.Source(mp.GaussianSource(frequency=frequency, fwidth=0.25 * frequency),
                       component=mp.Ex, center=position(-setup.source_distance),
                       size=mp.Vector3(setup.source_width, setup.source_width, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), dimensions=3, geometry=geometry,
                        boundary_layers=[mp.PML(setup.pml)], sources=[source],
                        resolution=resolution, Courant=COURANT)
    planes = {"obs": EXP.distance}
    if not with_screen:
        planes["ap"] = 0.0
    snapshots, audits = {}, {}
    geometry_audit, geometry_arrays = None, {}
    try:
        width = 2 * domain["monitor_half_width"]
        monitors = {
            key: sim.add_dft_fields([mp.Ex], frequency, 0, 1, center=position(z),
                                    size=mp.Vector3(width, width, 0))
            for key, z in planes.items()
        }
        for time in times:
            sim.run(until=time - sim.meep_time())
            snapshots[time] = {"actual_time": float(sim.meep_time())}
            if abs(snapshots[time]["actual_time"] - time) > COURANT / resolution + 1e-9:
                raise RuntimeError("Meep missed the requested time checkpoint")
            for key, z in planes.items():
                field, audit = dft_snapshot(sim, monitors[key], mp.Ex, archive["x" + key],
                                            archive["y" + key], z, center_z, resolution)
                if key in audits and audits[key] != audit:
                    raise RuntimeError("Meep DFT coordinates changed between checkpoints")
                snapshots[time][key] = field
                audits[key] = audit
            if mp.am_master():
                print(f"  saved DFT snapshot at t={sim.meep_time():.6g}", flush=True)
        if with_screen:
            geometry_audit, geometry_arrays = inspect_screen(mp, sim, center_z, resolution)
    finally:
        sim.reset_meep()
    return snapshots, audits, geometry_audit, geometry_arrays


def capture(mp, resolution, times, archive, metadata, fixed_mean, out):
    if mp.am_master():
        print(f"Resolution {resolution}: no-screen incident field", flush=True)
    incident, incident_axes, _, _ = simulate(mp, resolution, times, archive, False)
    if mp.am_master():
        print(f"Resolution {resolution}: PEC transmitted field", flush=True)
    screen, screen_axes, geometry, geometry_arrays = simulate(mp, resolution, times, archive, True)
    if incident_axes["obs"] != screen_axes["obs"]:
        raise RuntimeError("Incident and transmitted monitors do not share an observation grid")
    z = screen_axes["obs"]["actual_lab_z"]
    plane = ScalarPlane(wavelength=EXP.wavelength, fresnel_number=EXP.fresnel_number,
                        theta_ap=EXP.theta_ap, aspect_ratio=EXP.aspect_ratio, z=z)
    rs, fresnel = scalar_fields(plane, metadata["quadrature_order"], archive["xobs"], archive["yobs"])
    carrier = np.exp(1j * EXP.k * z)
    records = {}
    weights = archive["wobs"]
    for time in times:
        reference = incident[time]
        raw = screen[time]["obs"]
        mean, uniformity = incident_metrics(reference["ap"], archive["wap"])
        # Fixed archived drive calibration for comparisons. The old per-run
        # normalization is also saved to retain the existing NPZ contract.
        fixed = raw / (fixed_mean * carrier)
        record = {"resolution": resolution, "time": time,
                  "actual_incident_time": reference["actual_time"],
                  "actual_screen_time": screen[time]["actual_time"],
                  "observation_z": z, "observation_delta_z": z - EXP.distance,
                  **uniformity,
                  **comparison(fixed, rs, weights, "rs"),
                  **comparison(fixed, fresnel, weights, "fresnel")}
        records[time] = (record, raw, reference["ap"], reference["obs"])
        if mp.am_master():
            coords = {key: archive[key] for key in KEYS[:6]}
            grids = {
                f"{kind}_{key}_{axis}": value[f"native_meep_{axis}"]
                for kind, monitors in (("incident", incident_axes), ("screen", screen_axes))
                for key, value in monitors.items() for axis in "xyz"
            }
            np.savez_compressed(out / f"r{resolution}_t{time}_complex_fields.npz", **coords,
                                incident_aperture=reference["ap"], incident_observation=reference["obs"],
                                meep=raw / mean, meep_raw=raw, meep_fixed_incident=fixed,
                                rs=rs, fresnel=fresnel, observation_z_lab=z,
                                grid_center_z_lab=Setup().domain(resolution)["center_z"],
                                **grids)
    if mp.am_master():
        np.savez_compressed(out / f"r{resolution}_geometry_grid.npz", **geometry_arrays)
    return records, {"incident": incident_axes, "screen": screen_axes, "geometry": geometry}


def differences(left, right, archive, label):
    return {
        **comparison(left[1], right[1], archive["wobs"], label),
        **comparison(left[2], right[2], archive["wap"], f"{label}_incident_aperture"),
        **comparison(left[3], right[3], archive["wobs"], f"{label}_incident_observation"),
    }


def summary_rows(records, archive, temporal_pairs=None):
    rows = []
    for index, current in enumerate(records):
        # Compare each checkpoint to the final checkpoint and to its immediately
        # preceding checkpoint. The first point has no previous checkpoint.
        row = {**current[0], **differences(current, records[-1], archive, "vs_final")}
        if index:
            row.update(differences(current, records[index - 1], archive, "vs_previous"))
        if temporal_pairs is not None:
            row.update(differences(current, temporal_pairs[index], archive, "time_tail"))
            row["time_tail_from"] = temporal_pairs[index][0]["time"]
        rows.append(row)
    return rows


def save_sweep(out, name, rows, axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, panels = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
    for ax, quantity, unit in zip(panels, ("relative_l2", "phase_rms"), ("%", "rad")):
        for prefix, label in (("vs_final", "To final"), ("vs_previous", "To previous"),
                              ("time_tail", "Temporal tail")):
            suffix = "_aligned_rad" if quantity == "phase_rms" else "_raw"
            key = f"{prefix}_{quantity}{suffix}"
            selected = [row for row in rows if key in row]
            if selected:
                ax.plot([row[axis] for row in selected],
                        [(100 if unit == "%" else 1) * row[key] for row in selected],
                        "o-", label=label)
        ax.set_ylabel(f"{'Complex L2' if unit == '%' else 'Aligned phase RMS'} ({unit})")
        ax.grid(alpha=0.25)
        ax.legend()
    panels[-1].set(xlabel=axis.capitalize(), xticks=[row[axis] for row in rows])
    fig.savefig(out / f"{name}.png", dpi=170)
    plt.close(fig)


def assess(rows, spatial=False):
    keys = ["vs_previous_relative_l2_raw", "vs_previous_incident_aperture_relative_l2_raw",
            "vs_previous_incident_observation_relative_l2_raw"]
    last_pair = {key: rows[-1][key] for key in keys}
    if spatial:
        for key in keys:
            tail_key = key.replace("vs_previous", "time_tail")
            last_pair[tail_key] = max(row[tail_key] for row in rows[-2:])
    largest = max(last_pair.values())
    return {"status": ("quick_check_only" if len(rows) < 3 else
                       "last_pair_within_tolerance" if largest <= L2_TOLERANCE
                       else "last_pair_outside_tolerance"),
            "last_pair": last_pair, "max_relative_l2": largest,
            "threshold": L2_TOLERANCE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Validate and print cost without running Meep")
    parser.add_argument("--time-sweep", action="store_true", help="Run temporal convergence at resolution 6")
    parser.add_argument("--resolution-sweep", action="store_true", help="Compare resolutions at the final time")
    parser.add_argument("--quick", action="store_true", help="Use t=150/200 and resolutions 6/8")
    parser.add_argument("--out", type=Path, default=ROOT / "meep_numerical_results")
    args = parser.parse_args()
    baseline = ROOT / "meep_caseA_v2"
    archive, metadata = load_archive(baseline)
    verify_phase_alignment()
    fixed_mean, _ = incident_metrics(archive["incident_aperture"], archive["wap"])
    execute = (args.time_sweep or args.resolution_sweep) and not args.plan
    time_sweep = args.time_sweep or not args.resolution_sweep
    resolution_sweep = args.resolution_sweep or not args.time_sweep
    times, resolutions, jobs = make_plan(time_sweep, resolution_sweep, args.quick)
    costs = plan_cost(jobs)
    print("Archive and phase-alignment checks passed (144 observation points).")
    for cost in costs:
        low, high = cost["estimated_memory_gib_range"]
        print(f"r={cost['resolution']:>2}, t={cost['times']}: "
              f"{cost['yee_cells']:,} Yee cells; {low:.2f}-{high:.2f} GiB estimated per run; "
              f"{cost['fdtd_runs']} FDTD runs x {cost['steps_per_run']:,} steps; "
              f"{cost['estimated_cell_updates']:.3e} cell updates")
    print(f"Total: {sum(cost['fdtd_runs'] for cost in costs)} FDTD runs; "
          f"{sum(cost['estimated_cell_updates'] for cost in costs):.3e} cell updates.", flush=True)
    if not execute:
        print("Plan only. No FDTD started.")
        return
    if args.out.resolve() == baseline.resolve():
        parser.error("--out must differ from the archive directory")
    try:
        import meep as mp
    except ImportError as exc:
        parser.error(f"Meep FDTD bindings are required: {exc}")
    if not hasattr(mp, "Simulation"):
        parser.error("Installed meep module is not the FDTD package")
    if mp.am_master():
        args.out.mkdir(parents=True, exist_ok=True)
    fields, audits = {}, {}
    for resolution, checkpoints in sorted(jobs.items()):
        fields[resolution], audits[resolution] = capture(
            mp, resolution, checkpoints, archive, metadata, fixed_mean, args.out,
        )
    if not mp.am_master():
        return
    report = {
        "plan": costs, "monitors_and_geometry": audits, "quick": args.quick,
        "observation_points": archive["xobs"].size,
        "notes": [
            "Each incident/transmitted run accumulates the same DFT across all checkpoints.",
            "The monitor z is the collapsed interpolated plane; the native support is recorded when exposed by Meep.",
            "Scalar fields use the validated observation z and fixed physical aperture.",
            "Raw temporal/spatial comparisons use unscaled DFT fields; scalar metrics use one fixed archived incident mean.",
            "Memory assumes 64-128 bytes per estimated cell and excludes additional MPI/runtime overhead.",
            "Final pair is a numerical stability check for this setup, not a bound on boundary or PEC/vector-model error.",
        ],
    }
    if time_sweep:
        rows = summary_rows([fields[6][time] for time in times], archive)
        report["time_sweep"] = {"assessment": assess(rows), "rows": rows}
        save_sweep(args.out, "time_convergence", rows, "time")
    if resolution_sweep:
        rows = summary_rows([fields[r][times[-1]] for r in resolutions], archive,
                            [fields[r][times[-2]] for r in resolutions])
        assessment = assess(rows, spatial=True)
        assessment["geometry_raster_verified"] = all(
            audits[r]["geometry"]["raster_status"] == "detected" for r in resolutions
        )
        report["resolution_sweep"] = {"assessment": assessment, "rows": rows}
        save_sweep(args.out, "resolution_convergence", rows, "resolution")
    (args.out / "numerical_convergence_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Saved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
