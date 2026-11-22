"""One-factor Meep boundary and source validation for the archived Case A."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fresnel_shape_sensitivity import Experiment, aperture_rule, observation_rule
from meep_shape_diagnostic_v2 import phase_align, scalar_fields


ROOT = Path(__file__).resolve().parent
EXP = Experiment()
THICKNESS = 0.2
RUNTIME = 150.0
TAIL_WINDOW = 15.0
CONVERGENCE_THRESHOLD = 0.01
KEYS = ("xobs", "yobs", "wobs", "xap", "yap", "wap", "incident_aperture",
        "incident_observation", "meep", "rs", "fresnel")
SWEEPS = {
    "A": ("xy_clearance", (1.8, 3.8)),
    "B": ("pml", (1.5, 2.0)),
    "C": ("source_distance", (0.65, 0.5)),
    "D": ("source_width", (19.6, 17.6)),
}


@dataclass(frozen=True)
class Setup:
    name: str = "control"
    group: str = "control"
    xy_clearance: float = 0.8
    pml: float = 1.0
    source_distance: float = 0.8  # Distance from the screen midplane.
    source_width: float = 21.6

    def domain(self, resolution):
        half_xy = EXP.radius + self.xy_clearance
        # The inner z faces are fixed even while the source or PML changes.
        low, high = -1.6, EXP.distance + 0.8
        size = [2 * (half_xy + self.pml)] * 2 + [high - low + 2 * self.pml]
        monitor_half = EXP.radius + 0.4
        clearances = {
            "opening_xy": self.xy_clearance,
            "source_xy": half_xy - self.source_width / 2,
            "source_z": -self.source_distance - low,
            "screen_z": min(-THICKNESS / 2 - low, high - THICKNESS / 2),
            "source_to_screen_front": self.source_distance - THICKNESS / 2,
            "observation_z": high - EXP.distance,
            "monitor_xy": half_xy - monitor_half,
        }
        if any(clearances[key] <= 0 for key in clearances if key != "source_xy"):
            raise ValueError(f"{self.name}: overlapping geometry or monitor: {clearances}")
        if clearances["source_xy"] < -1e-10:
            raise ValueError(f"{self.name}: source extends into lateral PML")
        # The baseline sheet touches the inner PML face nominally. On a Yee
        # grid the boundary can be snapped, so this is a diagnostic case.
        return {
            "cell": size, "center_z": (low + high) / 2,
            "clearances": clearances, "monitor_half_width": monitor_half,
            "source_touches_lateral_pml": abs(clearances["source_xy"]) <= 1e-10,
            "estimated_yee_cells": int(np.prod(np.ceil(np.array(size) * resolution))),
            "fdtd_runs": 2,
        }


def load_archive(folder):
    metadata = json.loads((folder / "diagnostic_metrics.json").read_text(encoding="utf-8"))
    for key, value in {"resolution": 6, "p": 2, "Nf_eff": 1,
                       "theta_ap": 0.1, "physical_screen_thickness": THICKNESS}.items():
        if not np.isclose(metadata[key], value):
            raise ValueError(f"Archived {key} does not match Case A")
    with np.load(folder / "complex_fields.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in KEYS}
    expected = (*observation_rule(EXP, metadata["observation_order"]),
                *aperture_rule(2, EXP, metadata["quadrature_order"]))
    for key, values in zip(KEYS[:6], expected):
        np.testing.assert_allclose(data[key], values, rtol=1e-11, atol=1e-13)
    for key in KEYS[6:]:
        array = data[key]
        size = data["xap"].size if key == "incident_aperture" else data["xobs"].size
        if array.shape != (size,) or not np.iscomplexobj(array) or not np.isfinite(array).all():
            raise ValueError(f"Invalid archived {key} array")
    rs, fr = scalar_fields(EXP, metadata["quadrature_order"], data["xobs"], data["yobs"])
    for key, values in (("rs", rs), ("fresnel", fr)):
        np.testing.assert_allclose(data[key], values, rtol=1e-11, atol=1e-13)
    return data, metadata


def weighted_norm(field, weights):
    return float(np.sqrt(np.dot(weights, abs(field)**2)))


def relative_l2(field, reference, weights):
    denominator = weighted_norm(reference, weights)
    if denominator == 0:
        raise ValueError("Zero reference norm")
    return weighted_norm(field - reference, weights) / denominator


def incident_metrics(field, weights):
    weights = weights / weights.sum()
    mean = complex(np.dot(weights, field))
    if abs(mean) <= 1e-14:
        raise ValueError("Zero aperture incident mean")
    deviation = abs(field / mean - 1)
    return mean, {
        "incident_aperture_rms_complex_nonuniformity": weighted_norm(deviation, weights),
        "incident_aperture_max_complex_nonuniformity": float(deviation.max()),
        "incident_mean_real": mean.real, "incident_mean_imag": mean.imag,
        "incident_mean_amplitude": abs(mean), "incident_mean_phase_rad": float(np.angle(mean)),
        "incident_rms_amplitude": weighted_norm(field, weights),
    }


def comparison(field, reference, weights, prefix):
    overlap = np.vdot(reference, weights * field)
    phase_weights = weights * abs(reference)**2
    phase_error = np.angle(field * reference.conj())
    offset = np.angle(overlap)
    aligned_phase_error = np.angle(np.exp(1j * (phase_error - offset)))
    intensity = abs(field)**2
    ref_intensity = abs(reference)**2
    intensity_l1 = float(np.dot(weights, abs(intensity - ref_intensity)) / phase_weights.sum())
    return {
        f"{prefix}_relative_l2_raw": relative_l2(field, reference, weights),
        f"{prefix}_relative_l2_aligned": relative_l2(field * np.exp(-1j * offset),
                                                    reference, weights),
        f"{prefix}_intensity_l1": intensity_l1,  # Invariant under a global phase.
        f"{prefix}_phase_rms_raw_rad": weighted_norm(phase_error, phase_weights) / np.sqrt(phase_weights.sum()),
        f"{prefix}_phase_rms_aligned_rad": weighted_norm(aligned_phase_error, phase_weights) / np.sqrt(phase_weights.sum()),
        f"{prefix}_phase_offset_rad": float(offset),
    }


def sample_plane(sim, monitor, component, x, y, physical_z, grid_center, resolution):
    raw = np.asarray(sim.get_dft_array(monitor, component, 0)).squeeze()
    xs, ys, zs, _ = sim.get_array_metadata(dft_cell=monitor)
    xs, ys, zs = (np.asarray(axis).ravel() for axis in (xs, ys, zs))
    if raw.shape != (xs.size, ys.size) or min(xs.size, ys.size) < 2:
        raise RuntimeError(f"Unexpected Meep DFT grid shape {raw.shape}")
    for axis in (xs, ys):
        if not np.allclose(np.diff(axis), 1 / resolution, rtol=1e-6, atol=1e-9):
            raise RuntimeError("Meep XY grid does not match the specified resolution")
    if not zs.size or not np.all(np.isfinite(zs)) or np.max(abs(zs + grid_center - physical_z)) > 1 / resolution + 1e-9:
        raise RuntimeError(f"Meep monitor z metadata does not match requested z={physical_z:g}")
    sampled = RegularGridInterpolator((xs, ys), raw, bounds_error=True)(
        np.column_stack((x, y))
    )
    if not np.isfinite(sampled).all():
        raise RuntimeError("Nonfinite Meep field samples")
    return sampled, {
        "shape": list(raw.shape), "x_range": [float(xs[0]), float(xs[-1])],
        "y_range": [float(ys[0]), float(ys[-1])], "requested_z": physical_z,
        "metadata_z_lab": (zs + grid_center).tolist(),
    }


def simulate(mp, setup, domain, resolution, archive, with_screen):
    sx, sy, sz = domain["cell"]
    center = domain["center_z"]

    def position(z):
        return mp.Vector3(0, 0, z - center)

    geometry = []
    if with_screen:
        # Preserve an unbounded lateral screen: its PEC slab continues into
        # XY PML; the opening and both axial PEC faces remain in the interior.
        geometry = [
            mp.Block(size=mp.Vector3(sx, sy, THICKNESS), center=position(0), material=mp.metal),
            mp.Cylinder(radius=EXP.radius, height=THICKNESS + 2 / resolution,
                        axis=mp.Vector3(0, 0, 1), center=position(0), material=mp.air),
        ]
    frequency = 1 / EXP.wavelength
    source = mp.Source(mp.GaussianSource(frequency=frequency, fwidth=0.25 * frequency),
                       component=mp.Ex, center=position(-setup.source_distance),
                       size=mp.Vector3(setup.source_width, setup.source_width, 0))
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), dimensions=3,
                        boundary_layers=[mp.PML(setup.pml)], geometry=geometry,
                        sources=[source], resolution=resolution)
    planes = {"obs": (EXP.distance, archive["xobs"], archive["yobs"])}
    if not with_screen:
        planes["ap"] = (0.0, archive["xap"], archive["yap"])
    try:
        monitors = {}
        for key, (z, _, _) in planes.items():
            width = 2 * domain["monitor_half_width"]
            monitors[key] = sim.add_dft_fields([mp.Ex], frequency, 0, 1,
                                               center=position(z), size=mp.Vector3(width, width, 0))
        # Same integration window as the archive. The last increment is a
        # residual-field proxy that also detects other late arrivals.
        sim.run(until=RUNTIME - TAIL_WINDOW)
        early = {}
        for key, (z, x, y) in planes.items():
            early[key] = sample_plane(sim, monitors[key], mp.Ex, x, y, z, center, resolution)[0]
        sim.run(until=RUNTIME - sim.meep_time())
        results, audits, tails = {}, {}, {}
        for key, (z, x, y) in planes.items():
            results[key], audits[key] = sample_plane(sim, monitors[key], mp.Ex, x, y, z,
                                                     center, resolution)
            tails[key] = relative_l2(early[key], results[key], archive[f"w{key}"])
        return results, audits, tails
    finally:
        sim.reset_meep()


def setup_list(groups, quick):
    control = Setup()
    settings = [control]
    for group in groups:
        parameter, levels = SWEEPS[group]
        for index, value in enumerate(levels[:1] if quick else levels, start=1):
            settings.append(replace(control, name=f"{group}{index}", group=group,
                                    **{parameter: value}))
    return settings


def run_sweep(mp, settings, plans, resolution, archive, saved_mean, out):
    weights = archive["wobs"]
    baseline_raw = archive["meep"] * saved_mean
    rows, audits, raw_fields = [], {}, {}
    for setup, domain in zip(settings, plans):
        if mp.am_master():
            print(f"{setup.name}: no screen", flush=True)
        incident, incident_audit, incident_tail = simulate(
            mp, setup, domain, resolution, archive, with_screen=False,
        )
        if mp.am_master():
            print(f"{setup.name}: circular PEC aperture", flush=True)
        screen, screen_audit, screen_tail = simulate(
            mp, setup, domain, resolution, archive, with_screen=True,
        )
        raw = screen["obs"]
        raw_fields[setup.name] = raw
        mean, uniformity = incident_metrics(incident["ap"], archive["wap"])
        row = {
            **asdict(setup), "resolution": resolution,
            **domain["clearances"], **uniformity,
            "incident_mean_gain_vs_archive": abs(mean / saved_mean),
            "incident_observation_dft_tail_l2": incident_tail["obs"],
            "incident_aperture_dft_tail_l2": incident_tail["ap"],
            "transmitted_dft_tail_l2": screen_tail["obs"],
        }
        for prefix, test, ref in (
            ("baseline_meep", raw, baseline_raw),
            ("control_meep", raw, raw_fields["control"]),
            ("rs", raw / saved_mean, archive["rs"]),
            ("fresnel", raw / saved_mean, archive["fresnel"]),
            ("rs_per_incident", raw / mean, archive["rs"]),
            ("fresnel_per_incident", raw / mean, archive["fresnel"]),
        ):
            row.update(comparison(test, ref, weights, prefix))
        rows.append(row)
        audits[setup.name] = {"incident_no_screen": incident_audit, "transmitted": screen_audit}
        if mp.am_master():
            # Preserve the archive's legacy keys and per-run incident convention.
            data = {key: archive[key] for key in (*KEYS[:6], "rs", "fresnel")}
            np.savez_compressed(out / f"{setup.name}_complex_fields.npz", **data,
                                meep=raw / mean, meep_raw=raw, meep_fixed_incident=raw / saved_mean,
                                incident_aperture=incident["ap"], incident_observation=incident["obs"])
    return rows, audits, raw_fields


def assess(rows, fields, weights):
    findings = {}
    for group in SWEEPS:
        levels = [rows[0], *(row for row in rows if row["group"] == group)]
        deltas = []
        for a, b in zip(levels, levels[1:]):
            reference = fields[b["name"]]
            test = fields[a["name"]]
            deltas.append({"from": a["name"], "to": b["name"],
                           **comparison(test, reference, weights, "step")})
        if deltas:
            last_change = deltas[-1]["step_relative_l2_raw"]
            max_tail = max(row[key] for row in levels[-2:] for key in
                           ("transmitted_dft_tail_l2", "incident_aperture_dft_tail_l2",
                            "incident_observation_dft_tail_l2"))
            status = "source_sensitivity"
            if group in ("A", "B"):
                if len(deltas) < 2:
                    status = "insufficient_levels"
                elif max(last_change, max_tail) <= CONVERGENCE_THRESHOLD:
                    status = "last_pair_within_tolerance"
                else:
                    status = "not_stable_within_tolerance"
            findings[group] = {"status": status, "adjacent_steps": deltas,
                               "last_pair_max_tail_l2": max_tail}
    return findings


def plots(out, rows, archive, raw_fields, baseline_raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, filename):
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)

    for group, filename, xlabel in (
        ("A", "boundary_convergence.png", "Opening to XY PML clearance"),
        ("B", "pml_convergence.png", "PML thickness"),
    ):
        selected = [row for row in rows if row["group"] in ("control", group)]
        axis = SWEEPS[group][0]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for key, label in (
            ("control_meep_relative_l2_raw", "Change from live control"),
            ("rs_relative_l2_aligned", "RS-I residual, phase-aligned"),
            ("transmitted_dft_tail_l2", "Final-window transmitted DFT change"),
        ):
            ax.plot([r[axis] for r in selected], [100 * r[key] for r in selected], "o-", label=label)
        ax.set(xlabel=xlabel, ylabel="Relative L2 (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, filename)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, group, label in zip(axes, ("C", "D"), ("Source to screen midplane", "Source XY width")):
        axis = SWEEPS[group][0]
        selected = sorted((row for row in rows if row["group"] in ("control", group)),
                          key=lambda r: r[axis])
        for statistic in ("rms", "max"):
            ax.plot([r[axis] for r in selected],
                    [100 * r[f"incident_aperture_{statistic}_complex_nonuniformity"]
                     for r in selected], "o-", label=statistic.upper())
        ax.set(xlabel=label, ylabel="Incident complex nonuniformity (%)")
        ax.legend()
        ax.grid(alpha=0.3)
    save(fig, "source_uniformity.png")

    chosen = [rows[0], *(next(row for row in reversed(rows) if row["group"] == group)
                         for group in SWEEPS if any(row["group"] == group for row in rows))]
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.25 * len(chosen), 6),
                             squeeze=False, layout="constrained")
    scale = weighted_norm(baseline_raw, archive["wobs"])
    differences = []
    for row in chosen:
        raw = raw_fields[row["name"]]
        offset = row["baseline_meep_phase_offset_rad"]
        delta = (raw * np.exp(-1j * offset) - baseline_raw) / scale
        differences.append(delta)
    bound = max(1e-12, max(np.max(abs(delta)) for delta in differences))
    for col, (row, delta) in enumerate(zip(chosen, differences)):
        for axis, values, name in zip(axes[:, col], (delta.real, delta.imag), ("Re", "Im")):
            dots = axis.scatter(archive["xobs"], archive["yobs"], c=values, s=19,
                                cmap="RdBu_r", vmin=-bound, vmax=bound)
            axis.set(title=f"{row['name']} · {name}", xlabel="x", ylabel="y",
                     aspect="equal", xlim=(0, 5.1), ylim=(0, 5.1))
    fig.colorbar(dots, ax=axes.ravel().tolist(), label="Difference / archived field RMS")
    fig.suptitle("Phase-aligned complex difference from archived Meep")
    save(fig, "complex_field_difference.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print the validated simulation plan (default)")
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs without importing Meep")
    mode.add_argument("--run", action="store_true", help="Explicitly start the planned FDTD runs")
    parser.add_argument("--quick", action="store_true", help="Use one perturbation per selected factor")
    parser.add_argument("--groups", nargs="+", choices=SWEEPS, default=list(SWEEPS))
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--baseline", type=Path, default=ROOT / "meep_caseA_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "meep_boundary_results")
    args = parser.parse_args()
    if args.resolution < 6:
        parser.error("--resolution must be at least 6")
    if len(set(args.groups)) != len(args.groups):
        parser.error("--groups must not repeat a factor")
    archive, metadata = load_archive(args.baseline)
    mean, archive_incident = incident_metrics(archive["incident_aperture"], archive["wap"])
    baseline_metrics = dict(archive_incident)
    for label, key in (("RS", "rs"), ("Fresnel", "fresnel")):
        aligned = phase_align(archive["meep"], archive[key])
        value = relative_l2(aligned, archive[key], archive["wobs"])
        np.testing.assert_allclose(value, metadata[f"Meep_vs_{label}_phase_aligned"], rtol=1e-10)
        baseline_metrics[f"Meep_vs_{label}_phase_aligned"] = value
    np.testing.assert_allclose(
        archive_incident["incident_aperture_rms_complex_nonuniformity"],
        metadata["incident_aperture_rms_complex_nonuniformity"], rtol=1e-10,
    )
    settings = setup_list(args.groups, args.quick)
    plans = [setup.domain(args.resolution) for setup in settings]
    report = {
        "status": "validated_plan", "baseline": str(args.baseline.resolve()),
        "archived_metrics": baseline_metrics, "resolution": args.resolution,
        "observation_samples": archive["xobs"].size, "aperture_samples": archive["xap"].size,
        "wavelength": EXP.wavelength, "Nf_eff": EXP.fresnel_number,
        "theta_ap": EXP.theta_ap, "p": 2, "aperture_radius": EXP.radius,
        "observation_z": EXP.distance, "screen_thickness": THICKNESS,
        "runtime": RUNTIME, "late_dft_window": [RUNTIME - TAIL_WINDOW, RUNTIME],
        "plans": [{**asdict(setup), **plan} for setup, plan in zip(settings, plans)],
        "notes": [
            "The source and monitors remain outside PML by nominal geometry; the baseline sheet touches the lateral inner face.",
            "The PEC slab intentionally extends into lateral PML to preserve the archived screen geometry.",
            "Archived Meep raw amplitude is reconstructed using its saved aperture incident mean. Primary scalar comparisons use that single fixed complex calibration for all runs; per-run incident calibration is labeled separately.",
            "Intensity L1 is unchanged by phase alignment. Phase RMS uses reference intensity times quadrature weight.",
            "The late DFT increment is a residual proxy, not an isolated reflection coefficient.",
            "A/B stability uses the last two levels and tail checks at this resolution; one-factor changes do not decompose or bound vector/PEC model differences.",
        ],
        "last_pair_relative_l2_tolerance": CONVERGENCE_THRESHOLD,
    }
    print(f"Validated archive; {len(settings)} settings, {2 * len(settings)} FDTD runs.")
    for setup, plan in zip(settings, plans):
        clearance = plan["clearances"]
        print(f"{setup.name:7} {plan['estimated_yee_cells']:>12,} Yee cells/run  "
              f"XY opening/source clearance {clearance['opening_xy']:.2f}/"
              f"{clearance['source_xy']:.2f}  PML {setup.pml:g}  "
              f"source distance/width {setup.source_distance:g}/{setup.source_width:g}")
    if not args.run:
        print("No FDTD started. Use --run to execute this plan.")
        return
    if args.out.resolve() == args.baseline.resolve():
        parser.error("--out must differ from --baseline")
    try:
        import meep as mp
    except ImportError as exc:
        parser.error(f"Meep FDTD Python bindings are required: {exc}")
    if not hasattr(mp, "Simulation"):
        parser.error("Installed meep module is not the FDTD package")
    if mp.am_master():
        args.out.mkdir(parents=True, exist_ok=True)
    rows, audits, fields = run_sweep(mp, settings, plans, args.resolution, archive, mean, args.out)
    if mp.am_master():
        with (args.out / "boundary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plots(args.out, rows, archive, fields, archive["meep"] * mean)
        report.update(status="completed", metrics=rows, grid_audits=audits,
                      convergence=assess(rows, fields, archive["wobs"]))
        (args.out / "boundary_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"Saved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
