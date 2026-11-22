"""Compare Fresnel and scalar moment predictions with Meep Ex for a PEC aperture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fresnel_shape_sensitivity import (
    Experiment, SHAPES, fields, half_axes, moment_correction, observation_rule,
    plot_results, write_csv,
)


CASES = {"A": (1, 0.1), "B": (1, 0.2), "C": (1, 0.3),
         "D": (3, 0.1), "E": (3, 0.2)}
METRICS = ("E", "E_hat", "dE_dp", "dE_hat_dp")


@dataclass
class MeepReference:
    mp: object
    experiment: Experiment
    resolution: int
    thickness: float
    padding: float
    pml: float
    vertices: int
    decay: float

    def __post_init__(self):
        mp = self.mp
        exp = self.experiment
        self.frequency = 1 / exp.wavelength
        # Fixed domain for all p, including the wider diamond and FD stencils.
        ax, ay = half_axes(1, exp)
        margin = self.padding + self.pml
        self.sx, self.sy = 2 * (ax + margin), 2 * (ay + margin)
        self.source_z = -self.thickness - exp.wavelength
        lower = self.source_z - margin
        upper = exp.distance + margin
        self.sz = upper - lower
        self.z_center = (upper + lower) / 2
        width = 2 * (exp.roi_radius * exp.radius + 2 / self.resolution)
        self.monitor_size = mp.Vector3(width, width, 0)
        self.incident = None

    def geometry(self, p):
        mp = self.mp
        ax, ay = half_axes(p, self.experiment)
        phi = np.linspace(0, 2 * np.pi, self.vertices, endpoint=False)
        cosine, sine = np.cos(phi), np.sin(phi)
        edge = (abs(cosine)**p + abs(sine)**p)**(-1/p)
        x, y = ax * edge * cosine, ay * edge * sine
        polygon_area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
        scale = np.sqrt(self.experiment.area / polygon_area)
        # z=0 is the exit face. The air prism extends past both faces.
        z_screen = -self.thickness / 2 - self.z_center
        screen = mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.thickness),
                          center=mp.Vector3(0, 0, z_screen), material=mp.metal)
        aperture = mp.Prism(
            vertices=[mp.Vector3(u * scale, v * scale) for u, v in zip(x, y)],
            height=self.thickness + 2 / self.resolution,
            center=mp.Vector3(0, 0, z_screen), material=mp.air,
        )
        return [screen, aperture]

    def run(self, observation, p=None):
        mp = self.mp
        x, y, weights = observation
        sim = mp.Simulation(
            cell_size=mp.Vector3(self.sx, self.sy, self.sz),
            dimensions=3,
            boundary_layers=[mp.PML(self.pml)],
            symmetries=[mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y)],
            resolution=self.resolution,
            geometry=[] if p is None else self.geometry(p),
            sources=[mp.Source(
                mp.GaussianSource(self.frequency, fwidth=0.25 * self.frequency,
                                  is_integrated=True),
                component=mp.Ex, center=mp.Vector3(0, 0, self.source_z - self.z_center),
                size=mp.Vector3(self.sx, self.sy, 0),
            )],
        )
        try:
            dft = sim.add_dft_fields(
                [mp.Ex], self.frequency, 0, 1,
                center=mp.Vector3(0, 0, self.experiment.distance - self.z_center),
                size=self.monitor_size,
            )
            # Use a minimum post-source integration time for propagation across the cell.
            sim.run(until_after_sources=mp.stop_when_dft_decayed(
                tol=self.decay,
                minimum_run_time=self.experiment.distance + self.padding + 5,
            ))
            xs, ys, _, _ = sim.get_array_metadata(dft_cell=dft)
            raw = np.asarray(sim.get_dft_array(dft, mp.Ex, 0)).squeeze()
            if raw.shape != (len(xs), len(ys)):
                raise RuntimeError(f"Unexpected Meep DFT shape {raw.shape}")
            sampled = RegularGridInterpolator((xs, ys), raw)(np.column_stack((x, y)))
            if not np.isfinite(sampled).all():
                raise RuntimeError("Meep returned nonfinite Ex samples")
            if p is None:
                incident = np.dot(weights, sampled)
                if not abs(incident) > np.finfo(float).eps:
                    raise RuntimeError("Empty-cell incident field is too small")
                self.incident_nonuniformity = np.sqrt(
                    np.dot(weights, abs(sampled / incident - 1)**2)
                )
                return incident
            return sampled
        finally:
            sim.reset_meep()

    def evaluate(self, p, observation, scalar_order):
        if self.incident is None:
            self.incident = self.run(observation)
        x, y, weights = observation
        meep_field = self.run(observation, p) / self.incident
        scalar, *_ = fields(p, self.experiment, scalar_order, x, y)
        _, fresnel, m2, m4 = scalar.T
        correction = moment_correction(fresnel, m2, m4, self.experiment)
        corrected = fresnel + correction

        def norm(value):
            return np.sqrt(np.dot(weights, abs(value)**2))

        meep_norm = norm(meep_field)
        if not np.isfinite(meep_norm) or meep_norm == 0:
            raise RuntimeError("Invalid transmitted Meep field")
        return {
            "E": norm(meep_field - fresnel) / meep_norm,
            "E_hat": norm(correction) / norm(corrected),
            "moment_vs_meep": norm(meep_field - corrected) / meep_norm,
        }


def sweep(reference, exponents, step, scalar_order, observation_order):
    exp = reference.experiment
    observation = observation_rule(exp, observation_order)
    rows = []
    for p in exponents:
        center = reference.evaluate(p, observation, scalar_order)
        right = reference.evaluate(p + step, observation, scalar_order)
        if p == 1:
            far = reference.evaluate(p + 2 * step, observation, scalar_order)
            slopes = [(4 * (right[k] - center[k]) - (far[k] - center[k])) / (2 * step)
                      for k in METRICS[:2]]
        else:
            left = reference.evaluate(p - step, observation, scalar_order)
            slopes = [(right[k] - left[k]) / (2 * step) for k in METRICS[:2]]
        ax, ay = half_axes(p, exp)
        rows.append(dict(
            p=p, Nf_eff=exp.fresnel_number, theta_ap=exp.theta_ap,
            wavelength=exp.wavelength, AR=exp.aspect_ratio, a_eff=exp.radius,
            z=exp.distance, area=exp.area, a_x=ax, a_y=ay,
            resolution=reference.resolution, fd_step=step,
            incident_nonuniformity=reference.incident_nonuniformity,
            **center, dE_dp=slopes[0], dE_hat_dp=slopes[1],
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="A")
    parser.add_argument("--p", nargs="+", type=float, default=list(SHAPES))
    parser.add_argument("--resolutions", nargs="+", type=int, default=[12])
    parser.add_argument("--steps", nargs="+", type=float, default=[0.05])
    parser.add_argument("--screen-thickness", type=float, default=0.2)
    parser.add_argument("--padding", type=float, default=3.0)
    parser.add_argument("--pml", type=float, default=1.5)
    parser.add_argument("--vertices", type=int, default=256)
    parser.add_argument("--decay", type=float, default=1e-8)
    parser.add_argument("--scalar-order", type=int, default=48)
    parser.add_argument("--observation-order", type=int, default=16)
    parser.add_argument("--plan", action="store_true", help="Print domain size without loading Meep")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (any(not np.isfinite(p) or p < 1 for p in args.p)
            or sorted(set(args.p)) != args.p
            or any(not np.isfinite(h) or not 0 < h < 0.5 for h in args.steps)
            or sorted(set(args.steps), reverse=True) != args.steps
            or any(p != 1 and p - args.steps[0] < 1 for p in args.p)
            or any(v < 2 for v in (*args.resolutions, args.scalar_order, args.observation_order))
            or args.scalar_order % 2
            or sorted(set(args.resolutions)) != args.resolutions
            or args.vertices < 16 or args.vertices % 4
            or not all(np.isfinite(v) and v > 0
                       for v in (args.screen_thickness, args.padding, args.pml, args.decay))
            or args.padding < 2 / min(args.resolutions)):
        parser.error("Invalid geometry, quadrature, resolution or finite-difference settings")
    nf, theta = CASES[args.case]
    exp = Experiment(fresnel_number=nf, theta_ap=theta)
    if args.plan:
        from types import SimpleNamespace
        mp = SimpleNamespace(Vector3=lambda *xyz: xyz)
    else:
        try:
            import meep as mp
        except ImportError as exc:
            parser.error(f"Requires the FDTD Meep Python bindings: {exc}")
        if not hasattr(mp, "Simulation"):
            parser.error("Installed meep module is not the FDTD Meep package")

    runs = []
    for resolution in args.resolutions:
        reference = MeepReference(mp, exp, resolution, args.screen_thickness,
                                  args.padding, args.pml, args.vertices, args.decay)
        if args.plan:
            cells = np.prod(np.ceil(resolution * np.array(
                [reference.sx, reference.sy, reference.sz])).astype(np.int64))
            print(f"{args.case}: a_eff={exp.radius:g}, z={exp.distance:g}, "
                  f"resolution={resolution}, full domain Yee cells ~ {cells:,}, "
                  f"after mirror symmetry ~ {cells // 4:,}, "
                  f"FDTD runs={1 + 3 * len(args.p) * len(args.steps)}")
            continue
        for step in args.steps:
            # Each sweep creates its own observation rule with identical coordinates.
            rows = sweep(reference, args.p, step, args.scalar_order, args.observation_order)
            runs.extend(rows)
    if args.plan or not mp.am_master():
        return
    out = args.output_dir or Path(f"shape_sensitivity_meep_{args.case}")
    out.mkdir(parents=True, exist_ok=True)
    finest = runs[-len(args.p):]
    write_csv(out / "shape_sensitivity.csv", finest)
    changes = []
    for row in runs[:-len(args.p)]:
        target = finest[args.p.index(row["p"])]
        changes.append({**row, **{f"abs_change_{k}": abs(row[k] - target[k])
                                 for k in (*METRICS, "moment_vs_meep")}})
    if changes:
        write_csv(out / "convergence.csv", changes)
    plot_results(out / "shape_sensitivity.png", finest,
                 direct_label="Meep Ex vs Fresnel", predicted_label="Scalar moment prediction",
                 note=f"Case {args.case} · resolution {args.resolutions[-1]} · "
                      f"finite-difference step {args.steps[-1]:g}")
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
