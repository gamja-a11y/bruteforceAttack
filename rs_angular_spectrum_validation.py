"""Compare converged RS-I with exact scalar Helmholtz angular-spectrum propagation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from scipy.special import beta, betainc

from fresnel_quadrature_validation import (
    AREA, HALF_AXIS, NF, P, ROI_RADIUS, THETA, WAVELENGTH, Z,
    aperture_rule, gauss_fields, unit_legendre,
)
from fresnel_roi_convergence import polar_rule, verify_roi


K = 2*np.pi/WAVELENGTH
TARGET = 1e-3
PROBES = ((0., 0.), (1.13, .37), (2.5, 0.), (2., 2.), (3., 4.))


@dataclass(frozen=True)
class Case:
    name: str
    pitch: float
    base_width: float
    window_width: float
    sampling: str = "cell_average"
    interlace: int = 1

    @property
    def n(self):
        return int(round(self.window_width/self.pitch))

    @property
    def base_n(self):
        return int(round(self.base_width/self.pitch))

    @property
    def cache_key(self):
        # Once all nonzero cells fit, changing the zero-only base changes
        # nothing in an identically centered final FFT input.
        return self.pitch, self.n, self.sampling, self.interlace

    @property
    def effective_width(self):
        return self.window_width*self.interlace


def study_cases(quick, extended=False):
    window = 96. if quick else 144.
    pitches = (.15, .1, .075) if quick else (.15, .1, .075, .05, .0375)
    spans = (24., 48., 72., 96.) if quick else (24., 48., 72., 96., 144., 192.)
    base = 24.
    pitch = .1 if quick else .075
    studies = {
        "pitch": [Case(f"dx_{d:g}", d, base, window) for d in pitches],
        "domain": [Case(f"domain_{length:g}", pitch, length, length) for length in spans],
        "padding": [Case(f"padding_{length:g}", pitch, base, length) for length in spans],
        "boundary": [Case(f"edge_{kind}", pitch, base, window, kind)
                     for kind in ("point", "cell_average")],
        "matched_window": [Case(f"matched_{length:g}", pitch, length, window)
                           for length in (24., 48., 72.)],
    }
    if extended:
        studies["extended_window"] = [
            Case(f"effective_window_{144*m}", .075, 24., 144., interlace=m)
            for m in (2, 3, 4)
        ]
        studies["extended_window"].append(
            Case("effective_window_576_finer_pitch", .05, 24., 144., interlace=4),
        )
        studies["extended_window"].append(
            Case("effective_window_576_finest_pitch", .0375, 24., 96., interlace=6),
        )
    return studies


def validate_case(case):
    if (case.n < case.base_n or case.n % 2 or case.base_n % 2
            or not np.isclose(case.n*case.pitch, case.window_width, rtol=0, atol=1e-11)
            or not np.isclose(case.base_n*case.pitch, case.base_width, rtol=0, atol=1e-11)
            or case.base_width/2-case.pitch <= HALF_AXIS
            or case.window_width/2 < ROI_RADIUS+4*case.pitch):
        raise ValueError(f"Invalid FFT grid or truncated physical aperture: {case}")


def aperture_pixels(case):
    """Sample U0 by center or by exact superellipse/cell intersection area."""
    width = int(np.ceil(HALF_AXIS/case.pitch+0.5))
    grid = np.arange(-width, width+1)*case.pitch
    xy = abs(grid/HALF_AXIS)
    if case.sampling == "point":
        mask = (xy[:, None]**P+xy[None, :]**P <= 1).astype(float)
        return width, mask

    lower = np.maximum(0., xy-case.pitch/(2*HALF_AXIS))
    upper = xy+case.pitch/(2*HALF_AXIS)
    x0, x1 = lower[None, :], upper[None, :]
    y0, y1 = lower[:, None], upper[:, None]
    dimensionless_area = (x1-x0)*(y1-y0)

    def boundary(y):
        return np.maximum(0., 1-y**P)**(1/P)

    def primitive(x):
        return beta(1/P, 1+1/P)/P*betainc(1/P, 1+1/P, x**P)

    plateau_end = np.clip(boundary(y1), x0, x1)
    partial_end = np.clip(boundary(y0), plateau_end, x1)
    partial_end = np.minimum(partial_end, 1.)
    partial_start = np.minimum(plateau_end, 1.)
    area = ((plateau_end-x0)*(y1-y0)
            + primitive(partial_end)-primitive(partial_start)
            - y0*(partial_end-partial_start))
    if area.min() < -1e-12 or np.max(area-dimensionless_area) > 1e-12:
        raise RuntimeError("Invalid computed pixel/aperture intersection")
    mask = np.clip(area/dimensionless_area, 0, 1)
    return width, mask


def transfer(kt2, distance):
    # Choose Im(kz) >= 0 so evanescent components decay for z > 0.
    # The forward carrier exp(ikz) is removed, consistently with the RS-I code.
    propagating = kt2 <= K*K
    result = np.empty(kt2.shape, dtype=complex)
    t = kt2[propagating]
    result[propagating] = np.exp(-1j*distance*t/(np.sqrt(K*K-t)+K))
    t = kt2[~propagating]
    result[~propagating] = np.exp(-distance*np.sqrt(t-K*K)-1j*K*distance)
    return result


def propagate(initial, pitch, distance, cell_average=False, shift_x=0., shift_y=0.):
    """Input origin is at array [0,0]; return spectrum and physical frequencies."""
    spectrum = fft.fft2(initial)
    n = initial.shape[0]
    freq = fft.fftfreq(n, d=pitch)
    fx, fy = freq+shift_x, freq+shift_y
    kx2, ky2 = (2*np.pi*fx)**2, (2*np.pi*fy)**2
    # The piecewise-constant reconstruction of pixel averages has the
    # box basis Fourier transform, with a unit factor at DC.
    box_x = np.sinc(fx*pitch) if cell_average else np.ones(n)
    box_y = np.sinc(fy*pitch) if cell_average else np.ones(n)
    for start in range(0, n, 64):
        s = slice(start, start+64)
        spectrum[s] *= transfer(ky2[s, None]+kx2, distance)*box_y[s, None]*box_x
    return spectrum, fx, fy


def fft_probe_values(spectrum, fx, fy, probes):
    x, y = np.array(probes).T
    phase_x = np.exp(2j*np.pi*fx[:, None]*x)
    phase_y = np.exp(2j*np.pi*y[:, None]*fy)
    return np.sum(phase_y*(spectrum @ phase_x).T, axis=1)/spectrum.shape[0]**2


def sampled_solution(case, x, y, probe_ids):
    width, mask = aperture_pixels(case)
    n = case.n
    aperture_area = float(mask.sum()*case.pitch**2)
    if case.sampling == "cell_average":
        np.testing.assert_allclose(aperture_area, AREA, rtol=1e-10)
    indices = np.arange(-width, width+1) % n
    # Interpolation only needs a small central patch. Crop before spline filtering.
    half = int(np.ceil(ROI_RADIUS/case.pitch))+12
    patch_coords = np.arange(-half, half+1)*case.pitch
    patch_index = np.arange(-half, half+1) % n
    source_coords = np.arange(-width, width+1)*case.pitch
    local = np.zeros((patch_coords.size, patch_coords.size), dtype=complex)
    direct = np.zeros(len(probe_ids), dtype=complex)
    probes = [(x[i], y[i]) for i in probe_ids]
    # The wider padded DFT partitions exactly into offset frequency grids.
    shifts = np.arange(case.interlace)/case.effective_width
    for sy in shifts:
        for sx in shifts:
            shifted = bool(sx or sy)
            initial = np.zeros((n, n), dtype=complex if shifted else float)
            patch = mask*np.exp(-2j*np.pi*(sy*source_coords[:, None]+sx*source_coords)) if shifted else mask
            initial[np.ix_(indices, indices)] = patch
            spectrum, fx, fy = propagate(initial, case.pitch, Z, cell_average=True,
                                         shift_x=sx, shift_y=sy)
            del initial
            direct += fft_probe_values(spectrum, fx, fy, probes)
            spatial = fft.ifft2(spectrum, overwrite_x=True)
            del spectrum
            central = spatial[np.ix_(patch_index, patch_index)]
            if shifted:
                central *= np.exp(2j*np.pi*(sy*patch_coords[:, None]+sx*patch_coords))
            local += central
            del spatial
    local /= case.interlace**2
    direct /= case.interlace**2
    coords = [half+y/case.pitch, half+x/case.pitch]
    cubic = map_coordinates(local, coords, order=3, mode="mirror")
    linear = map_coordinates(local, coords, order=1, mode="mirror")
    return {
        "cubic": cubic, "linear": linear, "direct_probes": direct,
        "aperture_area": aperture_area,
        "aperture_area_relative_error": (aperture_area-AREA)/AREA,
    }


def metrics(test, reference, weights):
    intensity_weights = weights*abs(reference)**2
    denominator = intensity_weights.sum()
    offset = np.angle(np.vdot(reference, weights*test))
    phase = np.angle(test*reference.conj())

    def l2(field):
        return float(np.sqrt(np.dot(weights, abs(field-reference)**2)/denominator))

    def phase_rms(diff):
        return float(np.sqrt(np.dot(intensity_weights, diff*diff)/denominator))

    return {
        "raw_relative_complex_l2": l2(test),
        "phase_aligned_relative_complex_l2": l2(test*np.exp(-1j*offset)),
        "intensity_l1": float(np.dot(weights, abs(abs(test)**2-abs(reference)**2))/denominator),
        "weighted_phase_rms_raw_rad": phase_rms(phase),
        "weighted_phase_rms_aligned_rad": phase_rms(np.angle(np.exp(1j*(phase-offset)))),
        "phase_offset_rad": float(offset),
    }


def analytic_checks():
    # At r=0, substitute u=sqrt(z^2+rho^2) in the circular integral.
    # The radial kernel is the derivative of -z*exp(ik(u-z))/u.
    upper = np.hypot(Z, 10.)
    exact = 1-Z/upper*np.exp(1j*K*(upper-Z))
    nodes, weights = unit_legendre(64)
    t = (10*nodes/Z)**2
    root = np.sqrt(1+t)
    numerical = np.dot(weights*2*np.pi*100*nodes,
                       np.exp(1j*K*Z*t/(root+1))*(1+1j/(K*Z*root))/(1+t))/(1j*WAVELENGTH*Z)
    np.testing.assert_allclose(numerical, exact, atol=2e-12)
    pitch, n = .1, 32
    uniform = 1.2-.4j
    full_phase = np.exp(1j*K*Z)
    spectrum, _, _ = propagate(np.full((n, n), uniform), pitch, Z)
    plane = fft.ifft2(spectrum)
    np.testing.assert_allclose(plane*full_phase, uniform*full_phase, atol=1e-13)
    freq = 2*np.pi*12/(n*pitch)
    distance = .12
    evanescent = np.exp(1j*freq*pitch*np.arange(n))[None, :]*np.ones((n, 1))
    spectrum, _, _ = propagate(evanescent, pitch, distance)
    expected = np.exp(-distance*np.sqrt(freq**2-K*K)-1j*K*distance)
    np.testing.assert_allclose(fft.ifft2(spectrum), evanescent*expected, atol=2e-14)
    return {
        "circular_axis_analytic_real": float(exact.real), "circular_axis_analytic_imag": float(exact.imag),
        "circular_axis_abs_error": float(abs(exact-numerical)),
        "plane_wave_max_abs_error": float(np.max(abs(plane-uniform))),
        "evanescent_mode_abs_decay": float(abs(expected)),
    }


def plan(studies):
    unique = {}
    for cases in studies.values():
        for case in cases:
            validate_case(case)
            unique[case.cache_key] = case
    maximum = max(case.n for case in unique.values())
    peak = 64*maximum**2/2**20
    estimates = {"unique_fft_cases": len(unique), "largest_grid_n": maximum,
                 "approx_array_memory_mib": peak,
                 "total_fft_grid_points": sum(case.n**2*case.interlace**2 for case in unique.values()),
                 "fft_pairs": sum(case.interlace**2 for case in unique.values())}
    for group, cases in studies.items():
        print(f"{group}: "+", ".join(f"{c.name}={c.n}x{c.n}"
                                    + (f" x{c.interlace**2}" if c.interlace > 1 else "") for c in cases))
    print(f"{len(unique)} unique FFT cases; largest {maximum}x{maximum}; "
          f"~{peak:.0f} MiB array budget (excludes FFT library overhead).", flush=True)
    return estimates


def output_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(out, rows, profile_x, profile_reference, profile_fft):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, part, label in zip(axes, (np.real, np.imag), ("Real field", "Imaginary field")):
        ax.plot(profile_x, part(profile_reference), label="RS-I")
        ax.plot(profile_x, part(profile_fft), "--", label="ASM")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("x at y=0")
    fig.savefig(out/"complex_field_comparison.png", dpi=170)
    plt.close(fig)
    panels = [("pitch", "pitch", "Grid pitch"), ("domain", "window_width", "Unpadded domain width"),
              ("padding", "pad_each_side", "Zero padding per side")]
    if any(row["study"] == "extended_window" for row in rows):
        panels.append(("extended_window", "effective_window_width", "Effective period"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3*len(panels), 4), layout="constrained")
    for ax, (group, axis_key, label) in zip(axes, panels):
        data = [row for row in rows if row["study"] == group]
        for key, name in (("raw_relative_complex_l2", "Raw"),
                          ("phase_aligned_relative_complex_l2", "Phase-aligned")):
            ax.semilogy([r[axis_key] for r in data], [r[key] for r in data], "o-", label=name)
        ax.axhline(TARGET, color=".5", linestyle=":")
        ax.set(title=group.capitalize(), xlabel=label, ylabel="Relative complex L2")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out/"asm_convergence.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use the reduced FFT sweep")
    parser.add_argument("--extended", action="store_true", help="Add memory-bounded wide-window tests")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    studies = study_cases(args.quick, args.extended)
    cost = plan(studies)
    if args.plan or not (args.run or args.quick):
        print("Plan only. No propagation performed.")
        return
    label = "rs_asm_extended" if args.extended else "rs_asm_quick" if args.quick else "rs_asm_results"
    out = args.out or Path(__file__).resolve().parent / label
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new --out directory to preserve previous results")
    checks = analytic_checks()
    x, y, weights = polar_rule(32, 32)
    verify_roi(x, y, weights)
    roi_count = x.size
    profile = np.linspace(-ROI_RADIUS, ROI_RADIUS, 101)
    px, py = np.array(PROBES).T
    x = np.concatenate((x, px, profile))
    y = np.concatenate((y, py, np.zeros_like(profile)))
    probe_ids = np.arange(roi_count, roi_count+len(PROBES))
    profile_ids = slice(roi_count+len(PROBES), None)
    aperture = aperture_rule(64)
    if not aperture[3]["passed"]:
        raise RuntimeError("RS-I aperture quadrature audit failed")
    direct = gauss_fields(aperture, x, y)[0]
    results, rows, probe_rows = {}, [], []
    for study, cases in studies.items():
        previous = None
        for case in cases:
            if case.cache_key not in results:
                print(f"{study}: {case.name}, FFT {case.n}x{case.n}", flush=True)
                results[case.cache_key] = sampled_solution(case, x, y, probe_ids)
            result = results[case.cache_key]
            cubic, linear = result["cubic"], result["linear"]
            probes = result["direct_probes"]
            observed = slice(0, roi_count)
            uniform_probe_weights = np.full(len(probes), 1/len(probes))
            row = {
                "study": study, "case": case.name, "pitch": case.pitch, "n": case.n,
                "base_width": case.base_width, "window_width": case.window_width,
                "effective_window_width": case.effective_width, "interlace": case.interlace,
                "pad_each_side": (case.window_width-case.base_width)/2, "sampling": case.sampling,
                **metrics(cubic[observed], direct[observed], weights),
                "aperture_area": result["aperture_area"],
                "aperture_area_relative_error": result["aperture_area_relative_error"],
                "cubic_vs_linear_roi_l2": metrics(cubic[observed], linear[observed], weights)[
                    "raw_relative_complex_l2"],
                "cubic_vs_exact_discrete_fourier_probe_l2": metrics(
                    cubic[probe_ids], probes, uniform_probe_weights)["raw_relative_complex_l2"],
                "linear_vs_exact_discrete_fourier_probe_l2": metrics(
                    linear[probe_ids], probes, uniform_probe_weights)["raw_relative_complex_l2"],
                "spectral_probe_vs_rs_l2": metrics(probes, direct[probe_ids], uniform_probe_weights)[
                    "raw_relative_complex_l2"],
            }
            if previous is not None:
                row["change_from_previous_case_roi_l2"] = metrics(
                    cubic[observed], previous[0], weights)["raw_relative_complex_l2"]
                row["raw_l2_change_from_previous_case"] = row["raw_relative_complex_l2"]-previous[1]
            previous = (cubic[observed], row["raw_relative_complex_l2"])
            rows.append(row)
            for (xp, yp), i, exact_fft in zip(PROBES, probe_ids, probes):
                probe_rows.append({
                    "study": study, "case": case.name, "x": xp, "y": yp,
                    "rs_real": direct[i].real, "rs_imag": direct[i].imag,
                    "asm_real": cubic[i].real, "asm_imag": cubic[i].imag,
                    "asm_discrete_fourier_real": exact_fft.real,
                    "asm_discrete_fourier_imag": exact_fft.imag,
                })
    finest_case = studies["extended_window"][-1] if args.extended else studies["pitch"][-1]
    final = results[finest_case.cache_key]["cubic"]
    pitch = [row for row in rows if row["study"] == "pitch"]
    padding = [row for row in rows if row["study"] == "padding"]
    boundary = [row for row in rows if row["study"] == "boundary"]
    matched = [row for row in rows if row["study"] == "matched_window"]
    chosen = next(row for row in rows if row["case"] == finest_case.name)
    final_error = chosen["raw_relative_complex_l2"]
    report = {
        "status": "target_met" if final_error <= TARGET else "target_not_met",
        "mode": "extended" if args.extended else "quick" if args.quick else "full", "plan": cost,
        "conditions": {"nf": NF, "theta_ap": THETA, "ar": 1, "p": P, "a_eff": 10.,
                       "z": Z, "wavelength": WAVELENGTH, "roi_radius": ROI_RADIUS},
        "target_raw_complex_l2": TARGET, "analytic_checks": checks,
        "finest_pitch_case": pitch[-1],
        "comparison_case": chosen,
        "extended_window_cases": [row for row in rows if row["study"] == "extended_window"],
        "pitch_error_decreases_at_every_step": all(
            b["raw_relative_complex_l2"] <= a["raw_relative_complex_l2"] for a, b in zip(pitch, pitch[1:])
        ),
        "padding_final_step_roi_l2": padding[-1]["change_from_previous_case_roi_l2"],
        "boundary_sampling_change_roi_l2": boundary[-1]["change_from_previous_case_roi_l2"],
        "matched_padded_window_max_roi_l2_change": max(row.get("change_from_previous_case_roi_l2", 0.)
                                                        for row in matched),
        "notes": [
            "Both fields omit the same forward carrier exp(ikz). No fitted amplitude is used.",
            "The finite FFT span sets the period of the repeated aperture and the transfer-function "
            "frequency spacing. Window and padding sweeps probe those coupled effects; the matched-window "
            "control verifies that adding the same zeros before or during padding is equivalent.",
            "Center versus cell-average sampling changes only the pixel values. Both use "
            "the same rectangular pixel basis sinc and retain evanescent frequencies.",
            "Exact evaluation of the discrete Fourier series at the probes isolates interpolation error "
            "for that sampled spectrum. Cubic versus linear provides an ROI-wide interpolation diagnostic.",
            "Interlaced exact frequency grids evaluate wider padded windows with smaller FFT arrays.",
            "The error target concerns scalar half-space propagation with a prescribed aperture field.",
        ],
    }
    output_csv(out/"asm_convergence.csv", rows)
    output_csv(out/"complex_field_probes.csv", probe_rows)
    np.savez_compressed(out/"comparison_fields.npz", x=x[:roi_count], y=y[:roi_count], weights=weights,
                        rs=direct[:roi_count], asm=final[:roi_count])
    plot(out, rows, profile, direct[profile_ids], final[profile_ids])
    (out/"asm_validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(f"{report['status']}: raw relative L2={final_error:.6g}; saved to {out.resolve()}")


if __name__ == "__main__":
    main()
