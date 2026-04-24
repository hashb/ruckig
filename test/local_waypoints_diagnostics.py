"""Diagnostic tests for the local waypoints backend.

These tests intentionally focus on where the current implementation diverges
from the Kiemel & Kroeger 2024 behavior we want to reproduce:

1. Single-DoF traversal should stay on the 1D path almost exactly.
2. Multi-DoF tracking introduces the visible path deviation.
3. A paper-like 7-DoF geometric path currently remains far less accurate than
   the paper's reported geometric-shape results.

Run from the repository root after installing the editable package:

    .venv/bin/python test/local_waypoints_diagnostics.py
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np

from ruckig import InputParameter, OutputParameter, Result, Ruckig, WaypointsBackend


CONTROL_CYCLE = 0.01


@dataclass
class Metrics:
    duration: float
    max_dev: float
    mean_dev: float
    rms_dev: float
    p95_dev: float


def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-18:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab)) / len_sq, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def resample_positions_uniform(positions: np.ndarray, n_samples: int = 1500) -> np.ndarray:
    diffs = np.diff(positions, axis=0)
    chord = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(chord)])
    if arc[-1] < 1e-12:
        return positions
    s_uniform = np.linspace(0.0, arc[-1], n_samples)
    return np.column_stack([
        np.interp(s_uniform, arc, positions[:, d])
        for d in range(positions.shape[1])
    ])


def path_deviation(positions: np.ndarray, ref_poly: np.ndarray) -> tuple[float, float, float, float]:
    resampled = resample_positions_uniform(positions)
    deviations = np.array([
        min(point_to_segment_dist(p, a, b) for a, b in zip(ref_poly[:-1], ref_poly[1:]))
        for p in resampled
    ])
    return (
        float(deviations.max()),
        float(deviations.mean()),
        float(np.sqrt(np.mean(deviations**2))),
        float(np.percentile(deviations, 95)),
    )


def make_input(ref_poly: np.ndarray,
               max_velocity: list[float],
               max_acceleration: list[float],
               max_jerk: list[float]) -> InputParameter:
    dofs = ref_poly.shape[1]
    inp = InputParameter(dofs)
    inp.current_position = ref_poly[0].tolist()
    inp.current_velocity = [0.0] * dofs
    inp.current_acceleration = [0.0] * dofs
    inp.intermediate_positions = [p.tolist() for p in ref_poly[1:-1]]
    inp.target_position = ref_poly[-1].tolist()
    inp.target_velocity = [0.0] * dofs
    inp.target_acceleration = [0.0] * dofs
    inp.max_velocity = max_velocity
    inp.max_acceleration = max_acceleration
    inp.max_jerk = max_jerk
    return inp


def run_local(ref_poly: np.ndarray,
              max_velocity: list[float],
              max_acceleration: list[float],
              max_jerk: list[float]) -> Metrics:
    inp = make_input(ref_poly, max_velocity, max_acceleration, max_jerk)
    max_waypoints = len(inp.intermediate_positions)
    otg = Ruckig(inp.degrees_of_freedom, CONTROL_CYCLE, max_waypoints)
    otg.set_waypoints_backend(WaypointsBackend.Local)
    out = OutputParameter(inp.degrees_of_freedom, max_waypoints)

    outputs = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        outputs.append(copy(out))
        out.pass_to_input(inp)

    if res < 0:
        raise RuntimeError(f"local backend failed with code {int(res)}")

    positions = np.array([o.new_position for o in outputs])
    max_dev, mean_dev, rms_dev, p95_dev = path_deviation(positions, ref_poly)
    return Metrics(outputs[0].trajectory.duration, max_dev, mean_dev, rms_dev, p95_dev)


def resample_polyline(poly: np.ndarray, spacing: float) -> np.ndarray:
    diffs = np.diff(poly, axis=0)
    chord = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(chord)])
    total = arc[-1]
    n_segments = max(1, int(np.ceil(total / spacing)))
    s_uniform = np.linspace(0.0, total, n_segments + 1)
    return np.column_stack([
        np.interp(s_uniform, arc, poly[:, d])
        for d in range(poly.shape[1])
    ])


def paper_like_lemniscate(spacing: float = 0.2) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, 2001)
    dense = np.column_stack([
        0.72 * np.sin(t),
        0.52 * np.sin(t) * np.cos(t),
        0.42 * np.sin(t + 0.55),
        0.34 * np.sin(2.0 * t - 0.35),
        0.30 * np.cos(t + 0.8),
        0.24 * np.sin(3.0 * t + 0.2),
        0.20 * np.cos(2.0 * t - 0.7),
    ])
    return resample_polyline(dense, spacing)


def check_single_dof_is_accurate() -> Metrics:
    ref = np.array([[0.0], [1.0], [-0.5], [0.8], [0.0]])
    metrics = run_local(ref, [1.71], [10.0], [100.0])
    assert metrics.max_dev < 1e-6, metrics
    return metrics


def check_multi_dof_tracking_is_source_of_error() -> Metrics:
    ref = np.array([
        [0.0, 0.0],
        [1.0, 0.5],
        [-0.5, 0.6],
        [0.8, -0.4],
        [0.0, 0.0],
    ])
    metrics = run_local(ref, [1.71, 1.71], [10.0, 10.0], [100.0, 100.0])
    assert metrics.mean_dev > 0.02, metrics
    return metrics


def check_equal_axes_straight_line_is_accurate() -> Metrics:
    ref = np.linspace(np.zeros(3), np.ones(3), 5)
    metrics = run_local(ref, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0])
    assert metrics.max_dev < 1e-6, metrics
    return metrics


def check_scaled_straight_line_already_deviates() -> Metrics:
    ref = np.linspace(np.zeros(3), np.array([1.0, 0.8, 0.6]), 5)
    metrics = run_local(ref, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0])
    assert 0.02 < metrics.mean_dev < 0.05, metrics
    return metrics


def check_paper_like_shape_still_deviates() -> Metrics:
    ref = paper_like_lemniscate(0.2)
    metrics = run_local(
        ref,
        [1.71, 1.71, 1.74, 2.27, 2.44, 3.14, 3.14],
        [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
        [300.0, 150.0, 200.0, 250.0, 300.0, 400.0, 400.0],
    )
    assert metrics.mean_dev > 0.08, metrics
    assert metrics.max_dev > 0.15, metrics
    return metrics


def main() -> None:
    checks = [
        ("single_dof", check_single_dof_is_accurate),
        ("two_dof_tracking_gap", check_multi_dof_tracking_is_source_of_error),
        ("equal_axes_straight_line", check_equal_axes_straight_line_is_accurate),
        ("scaled_straight_line_gap", check_scaled_straight_line_already_deviates),
        ("paper_like_shape_gap", check_paper_like_shape_still_deviates),
    ]

    for name, check in checks:
        metrics = check()
        print(
            f"{name}: duration={metrics.duration:.4f} "
            f"max={metrics.max_dev:.6f} mean={metrics.mean_dev:.6f} "
            f"rms={metrics.rms_dev:.6f} p95={metrics.p95_dev:.6f}"
        )


if __name__ == "__main__":
    main()
