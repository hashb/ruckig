# Compare the local waypoints calculator and the cloud API (Ruckig Pro) side-by-side.
#
# Reports the trajectory duration, the kinematic-limit validity, and the path deviation from the
# piecewise-linear path through the waypoints (start -> intermediates -> target), and plots both
# trajectories. Cloud requests need network access; without it only the local backend is evaluated.
#
# Usage: python examples/18_waypoints_backend_compare.py [--problem 03|08] [--no-cloud]
#            [--global-steps N] [--local-steps N] [--smoothing-steps N] [--tolerance X]

import argparse
from copy import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ruckig import InputParameter, OutputParameter, Result, Ruckig, WaypointsBackend


def make_input(problem):
    inp = InputParameter(3)
    if problem == '08':
        inp.current_position = [0.8, 0.0, 0.5]
        inp.intermediate_positions = [
            [1.4, -1.6, 1.0],
            [-0.6, -0.5, 0.4],
            [-0.4, -0.35, 0.0],
            [-0.2, 0.35, -0.1],
            [0.2, 0.5, -0.1],
            [0.8, 1.8, -0.1],
        ]
        inp.target_position = [0.5, 1.2, 0.0]
        inp.max_velocity = [3.0, 2.0, 2.0]
        inp.max_acceleration = [6.0, 4.0, 4.0]
        inp.max_jerk = [16.0, 10.0, 20.0]
        return inp

    inp.current_position = [0.2, 0, -0.3]
    inp.current_velocity = [0, 0.2, 0]
    inp.current_acceleration = [0, 0.6, 0]
    inp.intermediate_positions = [
        [1.4, -1.6, 1.0],
        [-0.6, -0.5, 0.4],
        [-0.4, -0.35, 0.0],
        [0.8, 1.8, -0.1],
    ]
    inp.target_position = [0.5, 1, 0]
    inp.target_velocity = [0.2, 0, 0.3]
    inp.target_acceleration = [0, 0.1, -0.1]
    inp.max_velocity = [1, 2, 1]
    inp.max_acceleration = [3, 2, 2]
    inp.max_jerk = [6, 10, 20]
    return inp


def run_backend(backend, label, problem, args):
    otg = Ruckig(3, 0.01, 10)
    otg.set_waypoints_backend(backend)
    if backend == WaypointsBackend.Local:
        calculator = otg.calculator.local_waypoints_calculator
        if args.global_steps is not None:
            calculator.number_global_steps = args.global_steps
        if args.local_steps is not None:
            calculator.number_local_steps = args.local_steps
        if args.smoothing_steps is not None:
            calculator.number_smoothing_steps = args.smoothing_steps
        if args.tolerance is not None:
            calculator.smoothing_duration_tolerance = args.tolerance

    inp = make_input(problem)
    out = OutputParameter(3, 10)

    out_list = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        out_list.append(copy(out))
        out.pass_to_input(inp)

    if res < 0:
        raise RuntimeError(f'{label} backend failed with code {int(res)}')

    return otg, out_list


def stack(out_list, attr):
    return np.array([getattr(o, attr) for o in out_list])


# ---------------------------------------------------------------------------
# Path-deviation utilities (time-independent)
# ---------------------------------------------------------------------------

def build_reference_polyline(inp):
    pts = [inp.current_position] + list(inp.intermediate_positions) + [inp.target_position]
    return np.array(pts, dtype=float)


def path_deviations(positions, ref_poly):
    """Distance of each position to the nearest segment of the reference polyline."""
    dists = np.full(len(positions), np.inf)
    for seg in range(len(ref_poly) - 1):
        a, b = ref_poly[seg], ref_poly[seg + 1]
        ab = b - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-18:
            d = np.linalg.norm(positions - a, axis=1)
        else:
            t = np.clip(np.dot(positions - a, ab) / len_sq, 0.0, 1.0)
            d = np.linalg.norm(positions - (a + t[:, None] * ab), axis=1)
        dists = np.minimum(dists, d)
    return dists


def resample_positions_uniform(out_list, n_samples=2000):
    """Positions resampled at evenly-spaced arc-length fractions (time-independent comparison)."""
    pos = stack(out_list, 'new_position')
    chord = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(chord)])
    if arc[-1] < 1e-12:
        return pos
    s_uniform = np.linspace(0.0, arc[-1], n_samples)
    return np.column_stack([np.interp(s_uniform, arc, pos[:, d]) for d in range(pos.shape[1])])


def report_deviation(label, deviations):
    print(f'  {label:6s}  max={deviations.max():.6f}  mean={deviations.mean():.6f}  '
          f'rms={np.sqrt(np.mean(deviations**2)):.6f}  p95={np.percentile(deviations, 95):.6f}')


def compare_paths(runs, ref_poly, n_samples=2000):
    print('\n--- Path-deviation report (Euclidean distance to the piecewise-linear path) ---')
    resampled, dev_from_ref = {}, {}
    for run_key, (label, out_list) in runs.items():
        pos = resample_positions_uniform(out_list, n_samples)
        resampled[run_key] = pos
        dev_from_ref[run_key] = path_deviations(pos, ref_poly)
        report_deviation(label, dev_from_ref[run_key])
    return dev_from_ref, resampled


# ---------------------------------------------------------------------------
# Kinematic-validity check
# ---------------------------------------------------------------------------

def check_kinematic_limits(inp, runs, dt=1e-4, tol_factor=1.001):
    """Sample each trajectory finely and verify velocity, acceleration, and jerk limits."""
    print('\n--- Kinematic-limits validation ---')
    all_ok = True
    for run_key, (label, out_list) in runs.items():
        traj = out_list[0].trajectory
        times = np.arange(0.0, traj.duration, dt)
        vel = np.empty((len(times), 3))
        acc = np.empty((len(times), 3))
        for i, t in enumerate(times):
            _, v, a = traj.at_time(t)
            vel[i], acc[i] = v, a
        jerk = np.diff(acc, axis=0) / dt

        backend_ok = True
        for d in range(3):
            v_ok = np.max(np.abs(vel[:, d])) <= inp.max_velocity[d] * tol_factor
            a_ok = np.max(np.abs(acc[:, d])) <= inp.max_acceleration[d] * tol_factor
            j_ok = np.max(np.abs(jerk[:, d])) <= inp.max_jerk[d] * tol_factor
            backend_ok &= v_ok and a_ok and j_ok
        print(f'  {label:6s}  {"PASS" if backend_ok else "FAIL"}')
        all_ok &= backend_ok
    return all_ok


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_compare(filename, inp, runs):
    dofs = inp.degrees_of_freedom
    fig, axes = plt.subplots(dofs, 1, figsize=(9.0, 2.0 + 3.0 * dofs), dpi=120, sharex=True)
    colors = {'local': 'tab:blue', 'cloud': 'tab:orange'}

    for dof in range(dofs):
        ax = axes[dof]
        for run_key, (label, out_list) in runs.items():
            t = np.array([o.time for o in out_list])
            ax.plot(t, stack(out_list, 'new_position')[:, dof], color=colors[run_key], linestyle='-', label=f'{label} pos')
            ax.plot(t, stack(out_list, 'new_velocity')[:, dof], color=colors[run_key], linestyle='--', label=f'{label} vel', alpha=0.85)
            ax.plot(t, stack(out_list, 'new_acceleration')[:, dof], color=colors[run_key], linestyle=':', label=f'{label} acc', alpha=0.85)
        for wp in inp.intermediate_positions:
            ax.axhline(y=wp[dof], color='black', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.set_ylabel(f'DoF {dof + 1}')
        ax.grid(True, alpha=0.3)
        if dof == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

    axes[-1].set_xlabel('t [s]')
    fig.suptitle('Local waypoints calculator vs cloud API (Ruckig Pro)')
    fig.tight_layout()
    fig.savefig(filename)
    print(f'saved {filename}')


def plot_path_geometry(filename, ref_poly, runs, resampled, dev_from_ref):
    dofs = ref_poly.shape[1]
    colors = {'local': '#2196F3', 'cloud': '#FF9800'}
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref_poly, axis=0), axis=1))])
    ref_arc = arc / arc[-1] if arc[-1] > 1e-12 else arc

    fig, axes = plt.subplots(dofs + 1, 1, figsize=(11, 2.8 * (dofs + 1)), dpi=120, sharex=True)
    for d in range(dofs):
        ax = axes[d]
        ax.plot(ref_arc, ref_poly[:, d], color='black', linewidth=1.8, linestyle='--', zorder=5, label='reference (waypoints)')
        ax.scatter(ref_arc, ref_poly[:, d], color='black', s=40, zorder=6, marker='D')
        for run_key, pos in resampled.items():
            s = np.linspace(0.0, 1.0, len(pos))
            ax.plot(s, pos[:, d], color=colors[run_key], linewidth=1.5, alpha=0.9, zorder=4, label=runs[run_key][0])
            ax.fill_between(s, pos[:, d], np.interp(s, ref_arc, ref_poly[:, d]), color=colors[run_key], alpha=0.14, zorder=2)
        ax.set_ylabel(f'DoF {d + 1}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    ax = axes[dofs]
    for run_key, dev in dev_from_ref.items():
        s = np.linspace(0.0, 1.0, len(dev))
        ax.fill_between(s, dev, alpha=0.18, color=colors[run_key])
        ax.plot(s, dev, color=colors[run_key], linewidth=1.5, label=f'{runs[run_key][0]}  max={dev.max():.4f}  mean={dev.mean():.4f}')
    ax.set_ylabel('deviation')
    ax.set_xlabel('arc-length fraction (time-independent)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Joint path comparison: reference waypoints vs local/cloud backends')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f'saved {filename}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='03', choices=['03', '08'], help='example problem (03: example 03_waypoints, 08: example 08)')
    parser.add_argument('--no-cloud', action='store_true', help='skip the cloud API')
    parser.add_argument('--global-steps', type=int, default=None)
    parser.add_argument('--local-steps', type=int, default=None)
    parser.add_argument('--smoothing-steps', type=int, default=None)
    parser.add_argument('--tolerance', type=float, default=None, help='allowed relative duration increase in favor of a lower path deviation')
    args = parser.parse_args()

    runs = {}
    try:
        otg_local, out_local = run_backend(WaypointsBackend.Local, 'local', args.problem, args)
        print(f'local  duration = {out_local[0].trajectory.duration:0.4f} s  (calculation {out_local[0].calculation_duration / 1000:0.1f} ms)')
        runs['local'] = ('local', out_local)
    except Exception as e:
        print(f'[skip] local backend: {e}')

    if not args.no_cloud:
        try:
            otg_cloud, out_cloud = run_backend(WaypointsBackend.Cloud, 'cloud', args.problem, args)
            print(f'cloud  duration = {out_cloud[0].trajectory.duration:0.4f} s')
            runs['cloud'] = ('cloud', out_cloud)
        except Exception as e:
            print(f'[skip] cloud backend: {e}')

    if not runs:
        raise SystemExit('no backend produced a trajectory')

    inp_ref = make_input(args.problem)
    ref_poly = build_reference_polyline(inp_ref)

    check_kinematic_limits(inp_ref, runs)
    dev_from_ref, resampled = compare_paths(runs, ref_poly)

    out_dir = Path(__file__).parent.parent / 'build'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_compare(out_dir / f'18_backend_compare_{args.problem}.png', inp_ref, runs)
    plot_path_geometry(out_dir / f'18_path_geometry_{args.problem}.png', ref_poly, runs, resampled, dev_from_ref)
