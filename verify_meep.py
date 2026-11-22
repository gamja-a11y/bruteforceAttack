"""Unit checks and opt-in FDTD convergence checks for the pasted solver."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


FIELD_KEYS = ("U_RS", "U_Fr", "U_2D", "U_1D_sep", "U_Meep")
BASE_NAMES = ("BaseFieldSolver", "ApertureSpec", "ObservationGrid",
              "ScalarDiffractionSolver", "compute_error_metrics")


def load_target(path, base_module=None):
    if base_module:
        base = import_module(base_module)
        dependencies = {name: getattr(base, name) for name in BASE_NAMES}
    else:
        dependencies = {"BaseFieldSolver": object}
    return runpy.run_path(str(path), init_globals=dependencies)


def grid_fixture():
    x, y = np.meshgrid([-0.5, 0.0, 0.5], [-0.4, 0.4])
    return SimpleNamespace(
        normalized_coords=lambda: (x, y, None, None),
        hat_to_physical=lambda x_hat, y_hat, spec: (x_hat, y_hat),
    )


def unit_suite(target):
    solver_class = target["MeepFieldSolver"]
    cross_validate = target["cross_validate_point"]
    solver_globals = solver_class.propagate.__globals__

    class SolverTests(unittest.TestCase):
        def test_global_phase_preserves_amplitude(self):
            reference = np.array([[1 + 2j, -3j], [0, -2 + 1j]])
            test = 2.5 * reference * np.exp(0.73j)
            np.testing.assert_allclose(
                solver_class._align_global_phase(test, reference), 2.5 * reference,
                atol=1e-12,
            )

        def test_zero_or_orthogonal_overlap(self):
            reference = np.array([1j, 0j])
            for test in (np.zeros(2, dtype=complex), np.array([0j, 2 + 3j])):
                with self.subTest(test=test):
                    np.testing.assert_array_equal(
                        solver_class._align_global_phase(test, reference), test,
                    )

        def test_missing_meep(self):
            with patch.dict(solver_globals, {
                "mp": None, "_MEEP_IMPORT_ERROR": ImportError("test"),
            }):
                with self.assertRaises(ImportError):
                    solver_class()

        def test_wavelength_validation(self):
            with patch.dict(solver_globals, {"mp": SimpleNamespace(Ex="Ex")}):
                with self.assertRaises(ValueError):
                    solver_class().propagate(SimpleNamespace(wavelength=2.0), None)

        def test_geometry_dispatch(self):
            backend = Mock(Ex="Ex", metal="PEC")
            backend.Vector3.side_effect = lambda *xyz: xyz
            with patch.dict(solver_globals, {"mp": backend}):
                solver = solver_class(screen_thickness=0.25)
                for shape in ("rect", "ellipse"):
                    with self.subTest(shape=shape):
                        backend.reset_mock()
                        blocks = solver._build_aperture_blocks(
                            SimpleNamespace(shape=shape, a_x=0.75, a_y=0.5), 8, 6,
                        )
                        self.assertEqual(len(blocks), 2)
                        self.assertEqual(backend.Block.call_args_list[0].kwargs["size"],
                                         (8, 6, 0.25))
                        self.assertEqual(backend.Block.call_args_list[0].kwargs["material"],
                                         "PEC")
                        backend.Medium.assert_called_once_with(epsilon=1.0)
                        if shape == "rect":
                            aperture = backend.Block.call_args.kwargs
                            self.assertEqual(aperture["size"][:2], (1.5, 1.0))
                            self.assertGreaterEqual(aperture["size"][2], 0.25)
                with self.assertRaises(ValueError):
                    solver._build_aperture_blocks(SimpleNamespace(shape="invalid"), 8, 6)

        def test_ellipse_has_constant_cross_section_through_screen(self):
            backend = Mock(Ex="Ex", metal="PEC")
            backend.Vector3.side_effect = lambda *xyz: xyz
            spec = SimpleNamespace(shape="ellipse", a_x=0.75, a_y=0.5)
            with patch.dict(solver_globals, {"mp": backend}):
                solver_class(screen_thickness=0.25)._build_aperture_blocks(spec, 8, 6)
            self.assertFalse(
                backend.Ellipsoid.called,
                "An ellipsoid narrows toward each face; use a constant-section elliptical opening",
            )

        def test_comparison_direction_and_roi(self):
            grid = grid_fixture()
            fields = {key: np.full((2, 3), value, dtype=complex)
                      for value, key in enumerate(FIELD_KEYS, start=1)}
            metrics = Mock(side_effect=[{"id": 1}, {"id": 2}, {"id": 3}])
            solver = Mock()
            solver.propagate.return_value = fields
            spec = object()
            with patch.dict(cross_validate.__globals__, {"compute_error_metrics": metrics}):
                result = cross_validate(spec, grid, solver)
            solver.propagate.assert_called_once_with(spec, grid)
            self.assertEqual(result, {
                "Meep_vs_RS": {"id": 1}, "Fr_vs_Meep": {"id": 2}, "1D_vs_Meep": {"id": 3},
            })
            self.assertEqual(metrics.call_count, 3)
            comparisons = (("U_Meep", "U_RS"), ("U_Fr", "U_Meep"),
                           ("U_1D_sep", "U_Meep"))
            for call, (test, reference) in zip(metrics.call_args_list, comparisons):
                self.assertIs(call.args[0], fields[test])
                self.assertIs(call.args[1], fields[reference])
                np.testing.assert_array_equal(
                    call.kwargs["roi_mask"], [[False, True, False], [False, True, False]],
                )

        def test_propagation_axes_and_z_coordinates(self):
            backend = Mock(Ex="Ex", metal="PEC")
            backend.Vector3.side_effect = lambda *xyz: xyz
            sim = backend.Simulation.return_value
            grid = grid_fixture()
            x, y, _, _ = grid.normalized_coords()
            spec = SimpleNamespace(shape="rect", a_x=0.75, a_y=0.5,
                                   wavelength=1.0, distance=5.0)
            xs = np.array([-3.25, -0.25, 0.75, 3.0])
            ys = np.array([-2.75, 0.15, 2.8])

            def field(x_coord, y_coord):
                return 3 + 2 * x_coord + y_coord + 1j * (x_coord - 3 * y_coord)

            expected = field(x, y)
            raw = field(xs[:, None], ys[None, :]) * np.exp(0.73j)
            sim.get_dft_array.return_value = raw
            sim.get_array_metadata.return_value = xs, ys, [spec.distance], np.ones(raw.shape)
            scalar = Mock()
            scalar.return_value.propagate.return_value = {
                key: expected for key in FIELD_KEYS[:-1]
            }
            with patch.dict(solver_globals, {"mp": backend, "ScalarDiffractionSolver": scalar}):
                fields = solver_class().propagate(spec, grid)
            self.assertEqual(set(fields), set(FIELD_KEYS))
            source_z = backend.Source.call_args.kwargs["center"][2]
            screen_z = backend.Block.call_args_list[0].kwargs["center"][2]
            monitor_z = sim.add_dft_fields.call_args.kwargs["center"][2]
            with self.subTest(check="screen to monitor distance"):
                self.assertAlmostEqual(monitor_z - screen_z, spec.distance)
            with self.subTest(check="source to screen distance"):
                self.assertAlmostEqual(screen_z - source_z, 0.75)
            with self.subTest(check="DFT array metadata and complex interpolation"):
                sim.get_array_metadata.assert_called_once()
                np.testing.assert_allclose(fields["U_Meep"], expected, atol=1e-12)
            with self.subTest(check="decay evaluated after source shuts off"):
                self.assertIn("until_after_sources", sim.run.call_args.kwargs)

    return unittest.defaultTestLoader.loadTestsFromTestCase(SolverTests)


def relative_l2(test, reference):
    reference_norm = np.linalg.norm(reference)
    if not np.isfinite(reference_norm) or reference_norm == 0:
        raise ValueError("Reference field has a zero or nonfinite norm")
    return float(np.linalg.norm(test - reference) / reference_norm)


def check_fields(fields, expected_shape):
    for key in FIELD_KEYS:
        field = np.asarray(fields[key])
        if field.shape != expected_shape:
            raise ValueError(f"{key} shape {field.shape} != {expected_shape}")
        if not np.isfinite(field).all():
            raise ValueError(f"{key} contains nonfinite values")
        if not np.any(field):
            raise ValueError(f"{key} is identically zero")


def compare_fields(test, reference):
    # Fit one complex gain over the selected region to compare spatial shapes.
    norm = np.vdot(test, test).real
    if norm == 0:
        raise ValueError("Test field has zero energy in the comparison region")
    gain = np.vdot(test, reference) / norm
    return {
        "raw_complex_relative_l2": relative_l2(test, reference),
        "shape_complex_relative_l2": relative_l2(gain * test, reference),
        "gain_magnitude": float(abs(gain)),
    }


def run_convergence(target, make_case, resolutions, tolerance):
    report = {}
    all_passed = True
    for shape in ("rect", "ellipse"):
        spec, grid = make_case(shape)
        if spec.shape != shape:
            raise ValueError(f"Case factory returned {spec.shape!r} for {shape!r}")
        x, y, _, _ = grid.normalized_coords()
        roi = x**2 + y**2 <= 0.5**2
        if np.count_nonzero(roi) < 2:
            raise ValueError(f"{shape} ROI needs at least two grid points")
        previous = None
        rows = []
        for resolution in resolutions:
            fields = target["MeepFieldSolver"](resolution=resolution).propagate(spec, grid)
            check_fields(fields, x.shape)
            meep_roi = fields["U_Meep"][roi]
            row = {"resolution": resolution}
            for label, test, reference in (
                ("Meep_vs_RS", "U_Meep", "U_RS"),
                ("Fr_vs_Meep", "U_Fr", "U_Meep"),
                ("2D_vs_Meep", "U_2D", "U_Meep"),
                ("1D_vs_Meep", "U_1D_sep", "U_Meep"),
            ):
                row[label] = compare_fields(fields[test][roi], fields[reference][roi])
            if previous is not None:
                row["previous_vs_current"] = compare_fields(previous, meep_roi)
            previous = meep_roi
            rows.append(row)
        final_error = rows[-1]["previous_vs_current"]["shape_complex_relative_l2"]
        passed = final_error <= tolerance
        report[shape] = {"convergence_passed": passed, "runs": rows}
        all_passed &= passed
    return {"tolerance": tolerance, "passed": all_passed, "apertures": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the pasted solver (.txt or .py)")
    parser.add_argument("--base-module", help="Module providing the scalar solver and types")
    parser.add_argument("--integration", metavar="MODULE:FACTORY",
                        help="Run real Meep with a callable returning (spec, grid) for each shape")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[12, 16, 20])
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="Write the integration report as JSON")
    args = parser.parse_args()
    if args.integration:
        module, separator, factory = args.integration.partition(":")
        if not args.base_module or not separator or not module or not factory:
            parser.error("--integration requires MODULE:FACTORY and --base-module")
        if (len(args.resolutions) < 3 or min(args.resolutions) <= 0
                or sorted(set(args.resolutions)) != args.resolutions):
            parser.error("--resolutions requires at least three increasing positive integers")
        if not np.isfinite(args.tolerance) or args.tolerance <= 0:
            parser.error("--tolerance must be finite and positive")
    elif args.output:
        parser.error("--output requires --integration")
    # Allow project modules beside the source and in the invocation directory.
    sys.path[:0] = [str(args.source.resolve().parent), str(Path.cwd())]
    target = load_target(args.source, args.base_module)
    if not args.integration:
        result = unittest.TextTestRunner(verbosity=2).run(unit_suite(target))
        return int(not result.wasSuccessful())
    report = run_convergence(
        target, getattr(import_module(module), factory), args.resolutions, args.tolerance,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
