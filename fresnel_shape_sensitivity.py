"""Run the area-preserving superellipse diffraction experiment.

Conventions: a_eff = sqrt(A/pi), Nf = a_eff**2/(wavelength*z),
theta_ap = a_eff/z, AR = ax/ay. Uniform unit-amplitude illumination.
The comparison disk has radius 0.5*a_eff. All fields omit exp(i*k*z).

Let q = ((x-x')**2 + (y-y')**2)/z**2 and K = k*z.
M_(2j) = integral q**j exp(i*K*q/2) dA, for j = 0, 1, 2.
In units of the Fresnel prefactor, the RS-I kernel is
  exp(i*K*(sqrt(1+q)-1)) * (1+q)**(-1) * (1+i/(K*sqrt(1+q))).
Expanding its ratio to the Fresnel kernel through q**2 gives
  c0 = 1+i/K, c1 = -1-3i/(2K), c2 = 1+15i/(8K)-i*K*c0/8.
The predicted RS-I field uses c0*M0+c1*M2+c2*M4. Its predicted error
uses its own norm in the denominator and does not access the exact RS-I field.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import gammaln, roots_legendre


SHAPES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
RESULTS = ("E", "E_hat", "dE_dp", "dE_hat_dp")


@dataclass(frozen=True)
class Experiment:
    wavelength: float = 1.0
    fresnel_number: float = 1.0
    aspect_ratio: float = 1.0
    theta_ap: float = 0.1
    roi_radius: float = 0.5

    @property
    def radius(self):
        return self.fresnel_number * self.wavelength / self.theta_ap

    @property
    def distance(self):
        return self.radius / self.theta_ap

    @property
    def area(self):
        return np.pi * self.radius**2

    @property
    def k(self):
        return 2 * np.pi / self.wavelength


def half_axes(p, experiment):
    if not p >= 1:
        raise ValueError("p must be at least one")
    area_factor = 4 * np.exp(2 * gammaln(1 + 1/p) - gammaln(1 + 2/p))
    ay = np.sqrt(experiment.area / (area_factor * experiment.aspect_ratio))
    return experiment.aspect_ratio * ay, ay


@lru_cache(maxsize=None)
def unit_rule(order):
    nodes, weights = roots_legendre(order)
    return (nodes + 1) / 2, weights / 2


def aperture_rule(p, experiment, order):
    """Use elliptically scaled polar coordinates on a fixed integration domain."""
    ax, ay = half_axes(p, experiment)
    t, w = unit_rule(order)
    # Split at pi/4 to resolve the corner as p approaches the rectangular limit.
    angular_nodes, angular_weights = unit_rule(order // 2)
    angle = np.pi / 4 * np.concatenate((angular_nodes, 1 + angular_nodes))
    angular_weights = np.tile(angular_weights * np.pi / 4, 2)
    cosine, sine = np.cos(angle), np.sin(angle)
    if np.isinf(p):
        edge = 1 / np.maximum(cosine, sine)
    else:
        edge = (cosine**p + sine**p)**(-1/p)
    x = ax * np.outer(t, edge * cosine).ravel()
    y = ay * np.outer(t, edge * sine).ravel()
    weights = ax * ay * np.outer(w * t, angular_weights * edge**2).ravel()
    return (np.concatenate((x, -x, -x, x)), np.concatenate((y, y, -y, -y)),
            np.tile(weights, 4))


def observation_rule(experiment, order):
    # Reflection symmetry lets this rule represent the entire observation disk.
    t, w = unit_rule(order)
    radius = t * experiment.roi_radius * experiment.radius
    angle = t * np.pi / 2
    weights = np.outer(w * t, w).ravel()
    return (
        np.outer(radius, np.cos(angle)).ravel(),
        np.outer(radius, np.sin(angle)).ravel(),
        weights / weights.sum(),
    )


def fields(p, experiment, order, x, y):
    source_x, source_y, weights = aperture_rule(p, experiment, order)
    z = experiment.distance
    wave_distance = experiment.k * z
    prefactor = 1 / (1j * experiment.wavelength * z)
    chunks = []
    rs_seconds = moment_seconds = 0.0
    batch = max(1, min(32, 300_000 // weights.size))
    for start in range(0, x.size, batch):
        selection = slice(start, start + batch)
        q = ((x[selection, None] - source_x)**2
             + (y[selection, None] - source_y)**2) / z**2
        # q/(sqrt(1+q)+1) avoids cancellation near the optical axis.
        tick = perf_counter()
        root = np.sqrt(1 + q)
        rs_kernel = np.exp(1j * wave_distance * q / (root + 1))
        rs_kernel *= (1 + 1j / (wave_distance * root)) / (1 + q)
        rs = np.sum(rs_kernel * weights, axis=1)
        rs_seconds += perf_counter() - tick

        tick = perf_counter()
        weighted_fresnel = np.exp(0.5j * wave_distance * q) * weights
        m0 = np.sum(weighted_fresnel, axis=1)
        weighted_second = q * weighted_fresnel
        m2 = np.sum(weighted_second, axis=1)
        m4 = np.sum(q * weighted_second, axis=1)
        moment_seconds += perf_counter() - tick
        chunks.append(np.column_stack((rs, m0, m2, m4)))
    values = np.concatenate(chunks) * prefactor
    return values, weights.sum(), rs_seconds, moment_seconds


def evaluate(p, experiment, order, observation):
    x, y, weights = observation
    values, area, rs_time, moment_time = fields(p, experiment, order, x, y)
    rs, fresnel, m2, m4 = values.T
    K = experiment.k * experiment.distance
    c0 = 1 + 1j / K
    c1 = -1 - 1.5j / K
    c2 = 1 + 15j / (8 * K) - 1j * K * c0 / 8
    predicted_correction = (c0 - 1) * fresnel + c1 * m2 + c2 * m4
    predicted = fresnel + predicted_correction

    def norm(field):
        return np.sqrt(np.dot(weights, np.abs(field)**2))

    rs_norm = norm(rs)
    residual = rs - predicted
    return {
        "E": norm(rs - fresnel) / rs_norm,
        "E_hat": norm(predicted_correction) / norm(predicted),
        "corrected_field_error": norm(residual) / rs_norm,
        "area_relative_error": (area - experiment.area) / experiment.area,
        "rs_kernel_seconds": rs_time,
        "moment_kernel_seconds": moment_time,
    }


def sweep(experiment, aperture_order, observation_order, step):
    observation = observation_rule(experiment, observation_order)
    rows = []
    for p in SHAPES:
        center = evaluate(p, experiment, aperture_order, observation)
        right = evaluate(p + step, experiment, aperture_order, observation)
        if p == 1:
            far_right = evaluate(p + 2 * step, experiment, aperture_order, observation)
            slopes = [(4 * (right[key] - center[key]) - (far_right[key] - center[key]))
                      / (2 * step) for key in RESULTS[:2]]
        else:
            left = evaluate(p - step, experiment, aperture_order, observation)
            slopes = [(right[key] - left[key]) / (2 * step) for key in RESULTS[:2]]
        ax, ay = half_axes(p, experiment)
        rows.append({
            "p": p, "a_x": ax, "a_y": ay, "area_target": experiment.area,
            "a_eff": experiment.radius, "z": experiment.distance,
            "wavelength": experiment.wavelength, "Nf_eff": experiment.fresnel_number,
            "AR": experiment.aspect_ratio, "theta_ap": experiment.theta_ap,
            "roi_radius_a_eff": experiment.roi_radius,
            "aperture_order": aperture_order, "observation_order": observation_order,
            "fd_step": step, **center, "dE_dp": slopes[0], "dE_hat_dp": slopes[1],
        })
    return rows


def verify_analytic_cases(experiment, order):
    for p, factor in ((1, 2), (2, np.pi), (np.inf, 4)):
        ax, ay = half_axes(p, experiment)
        np.testing.assert_allclose(factor * ax * ay, experiment.area, rtol=1e-13)
        weights = aperture_rule(p, experiment, order)[2]
        np.testing.assert_allclose(weights.sum(), experiment.area, rtol=1e-10)

    axis = np.zeros(1)
    values = fields(2, experiment, order, axis, axis)[0]
    z, a, k = experiment.distance, experiment.radius, experiment.k
    radius = np.hypot(z, a)
    exact_rs = 1 - z / radius * np.exp(1j * k * a**2 / (radius + z))
    exact_fresnel = -np.expm1(0.5j * k * a**2 / z)
    np.testing.assert_allclose(values[0, :2], [exact_rs, exact_fresnel], atol=1e-11)


def convergence_checks(experiment, aperture_orders, observation_orders, steps):
    finest_aperture, finest_observation, finest_step = (
        aperture_orders[-1], observation_orders[-1], steps[-1]
    )
    final = sweep(experiment, finest_aperture, finest_observation, finest_step)
    settings = [
        *(("aperture", order, finest_observation, finest_step)
          for order in aperture_orders[:-1]),
        *(("observation", finest_aperture, order, finest_step)
          for order in observation_orders[:-1]),
        *(("finite_difference", finest_aperture, finest_observation, step)
          for step in steps[:-1]),
    ]
    checks = []
    for variable, aperture, observation, step in settings:
        rows = sweep(experiment, aperture, observation, step)
        for row, reference in zip(rows, final):
            changes = {f"abs_change_{key}": abs(row[key] - reference[key])
                       for key in RESULTS}
            checks.append({"varied": variable, **row, **changes})
    return final, checks


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    p = [row["p"] for row in rows]
    for ax, pair, ylabel in zip(
        axes, (RESULTS[:2], RESULTS[2:]), ("Relative field error (%)", "Sensitivity (% per unit p)")
    ):
        ax.plot(p, [100 * row[pair[0]] for row in rows], "o-", color="#174666",
                label="Direct RS-I vs Fresnel")
        ax.plot(p, [100 * row[pair[1]] for row in rows], "s--", color="#b05227",
                markerfacecolor="white", label="Complex-moment prediction")
        ax.grid(alpha=0.23)
        ax.set_ylabel(ylabel)
    axes[0].legend()
    axes[1].axhline(0, color="0.5", linewidth=0.8)
    axes[1].set(xlabel="Superellipse exponent p", xticks=p, xticklabels=[f"{v:g}" for v in p])
    fig.suptitle("Area-preserving superellipse diffraction")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("shape_sensitivity_results"))
    args = parser.parse_args()
    experiment = Experiment()
    verify_analytic_cases(experiment, 40)
    finest, convergence = convergence_checks(experiment, (12, 20, 32), (8, 12, 16),
                                              (0.02, 0.01, 0.005))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "shape_sensitivity.csv", finest)
    write_csv(args.output_dir / "convergence.csv", convergence)
    plot_results(args.output_dir / "shape_sensitivity.png", finest)
    for varied in ("aperture", "observation", "finite_difference"):
        # Compare the second-finest run with the finest for each refinement.
        group = [row for row in convergence if row["varied"] == varied][-len(SHAPES):]
        print(varied, ", ".join(
            f"{key}={max(row[f'abs_change_{key}'] for row in group):.3e}"
            for key in RESULTS
        ))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
