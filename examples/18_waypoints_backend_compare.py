# Compare the local (Kiemel & Kroeger 2024) and cloud (Ruckig Pro) waypoints
# backends side-by-side, plotting both trajectories on the same axes.
#
# Additionally reports (and plots) how much each backend's position path
# deviates from the "original path" – the piecewise-linear polyline through
# the waypoints (start → intermediates → target) – ignoring time.
#
# Requires that the Python module was built with both BUILD_LOCAL_WAYPOINTS
# and BUILD_CLOUD_CLIENT enabled. Cloud requests need network access.

from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ruckig import InputParameter, OutputParameter, Result, Ruckig, WaypointsBackend


def make_input():
    inp = InputParameter(3)
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


def run_backend(backend, label):
    otg = Ruckig(3, 0.01, 10)
    otg.set_waypoints_backend(backend)

    inp = make_input()
    out = OutputParameter(3, 10)

    out_list = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        out_list.append(copy(out))
        out.pass_to_input(inp)

    if res < 0:
        raise RuntimeError(f"{label} backend failed with code {int(res)}")

    return otg, out_list


def stack(out_list, attr):
    return np.array([getattr(o, attr) for o in out_list])


# ---------------------------------------------------------------------------
# Path-deviation utilities (time-independent)
# ---------------------------------------------------------------------------

def build_reference_polyline(inp):
    """Return (N, D) array of the piecewise-linear reference waypoints."""
    pts = (
        [inp.current_position]
        + list(inp.intermediate_positions)
        + [inp.target_position]
    )
    return np.array(pts, dtype=float)


def point_to_segment_dist(p, a, b):
    """Signed-length-independent distance from point p to segment [a, b].

    Returns the minimum Euclidean distance from p to the closest point on the
    finite segment a→b.
    """
    ab = b - a
    len_sq = np.dot(ab, ab)
    if len_sq < 1e-18:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / len_sq, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def path_deviations(positions, ref_poly):
    """For each row in *positions* compute the distance to the nearest segment
    of *ref_poly*.

    Parameters
    ----------
    positions : (T, D) array  – sampled trajectory positions
    ref_poly  : (N, D) array  – reference piecewise-linear waypoints

    Returns
    -------
    deviations : (T,) array of non-negative distances
    """
    n_segs = len(ref_poly) - 1
    dists = np.full(len(positions), np.inf)
    for seg in range(n_segs):
        a, b = ref_poly[seg], ref_poly[seg + 1]
        d = np.array([point_to_segment_dist(p, a, b) for p in positions])
        dists = np.minimum(dists, d)
    return dists


def path_deviations_per_axis(positions, ref_poly):
    """For each row in *positions* compute the per-axis absolute distance to
    the nearest point on *ref_poly*.

    Parameters
    ----------
    positions : (T, D) array  – sampled trajectory positions
    ref_poly  : (N, D) array  – reference piecewise-linear waypoints

    Returns
    -------
    axis_devs : (T, D) array of absolute per-axis deviations
    """
    n_segs = len(ref_poly) - 1
    best_dist_sq = np.full(len(positions), np.inf)
    closest = np.empty_like(positions)
    for seg in range(n_segs):
        a, b = ref_poly[seg], ref_poly[seg + 1]
        ab = b - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-18:
            cand = np.tile(a, (len(positions), 1))
        else:
            t = np.clip(np.dot(positions - a, ab) / len_sq, 0.0, 1.0)
            cand = a + t[:, None] * ab
        dist_sq = np.sum((positions - cand) ** 2, axis=1)
        mask = dist_sq < best_dist_sq
        best_dist_sq[mask] = dist_sq[mask]
        closest[mask] = cand[mask]
    return np.abs(positions - closest)


def resample_positions_uniform(out_list, n_samples=1000):
    """Return positions resampled at *n_samples* evenly-spaced arc-length
    fractions so that comparison is time-independent.

    The raw discrete positions are first accumulated into an arc-length
    parameterisation; we then interpolate at uniform arc-length steps.
    """
    pos = stack(out_list, "new_position")  # (T, D)
    # cumulative chord lengths
    diffs = np.diff(pos, axis=0)
    chord = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(chord)])
    total_arc = arc[-1]
    if total_arc < 1e-12:
        return pos  # degenerate – return as-is

    s_uniform = np.linspace(0.0, total_arc, n_samples)
    # Interpolate each DOF independently
    resampled = np.column_stack(
        [np.interp(s_uniform, arc, pos[:, d]) for d in range(pos.shape[1])]
    )
    return resampled


def report_deviation(label, deviations):
    print(
        f"  {label:6s}  max={deviations.max():.6f}  "
        f"mean={deviations.mean():.6f}  "
        f"rms={np.sqrt(np.mean(deviations**2)):.6f}  "
        f"p95={np.percentile(deviations, 95):.6f}"
    )


def compare_paths(runs, ref_poly, n_samples=2000):
    """Compute and print per-backend deviation from the reference polyline,
    and (if both backends are present) the path deviation between them.

    Returns a dict mapping run_key → deviations array (against ref_poly).
    """
    print("\n--- Path-deviation report (Euclidean distance, units match input) ---")
    print(f"Reference polyline: {len(ref_poly)} waypoints, "
          f"{len(ref_poly) - 1} segments")
    print(f"Sampling: {n_samples} arc-length-uniform points per trajectory\n")

    header = f"{'backend':6s}  {'max':>10s}  {'mean':>10s}  {'rms':>10s}  {'p95':>10s}"
    print(header)
    print("-" * len(header))

    axis_labels = [f"ax{d+1}" for d in range(ref_poly.shape[1])]

    resampled = {}
    dev_from_ref = {}
    for run_key, (label, out_list) in runs.items():
        pos = resample_positions_uniform(out_list, n_samples)
        resampled[run_key] = pos
        dev = path_deviations(pos, ref_poly)
        dev_from_ref[run_key] = dev
        report_deviation(label, dev)
        axis_devs = path_deviations_per_axis(pos, ref_poly)
        for d, ax_label in enumerate(axis_labels):
            report_deviation(f"  {label}/{ax_label}", axis_devs[:, d])

    # Cross-backend comparison (if both present, use the shorter resampled path)
    keys = list(runs.keys())
    if len(keys) == 2:
        pos_a = resampled[keys[0]]
        pos_b = resampled[keys[1]]
        # Align lengths
        n = min(len(pos_a), len(pos_b))
        cross_dev = np.linalg.norm(pos_a[:n] - pos_b[:n], axis=1)
        print()
        print(f"Cross-backend path deviation ({keys[0]} vs {keys[1]}):")
        report_deviation("cross", cross_dev)

    return dev_from_ref, resampled


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_compare(filename, inp_template, runs):
    dofs = inp_template.degrees_of_freedom
    fig, axes = plt.subplots(dofs, 1, figsize=(9.0, 2.0 + 3.0 * dofs), dpi=120, sharex=True)
    if dofs == 1:
        axes = [axes]

    colors = {"local": "tab:blue", "cloud": "tab:orange"}
    styles = {"position": "-", "velocity": "--", "acceleration": ":"}

    for dof in range(dofs):
        ax = axes[dof]
        for run_key, (label, out_list) in runs.items():
            t = np.array([o.time for o in out_list])
            p = stack(out_list, "new_position")[:, dof]
            v = stack(out_list, "new_velocity")[:, dof]
            a = stack(out_list, "new_acceleration")[:, dof]

            color = colors[run_key]
            ax.plot(t, p, color=color, linestyle=styles["position"], label=f"{label} pos")
            ax.plot(t, v, color=color, linestyle=styles["velocity"], label=f"{label} vel", alpha=0.85)
            ax.plot(t, a, color=color, linestyle=styles["acceleration"], label=f"{label} acc", alpha=0.85)

        # Intermediate waypoint markers (target positions for this DoF)
        wp_y = [wp[dof] for wp in inp_template.intermediate_positions]
        ax.scatter([], [], color="black", marker="x", label="waypoint")
        for y in wp_y:
            ax.axhline(y=y, color="black", linestyle=":", linewidth=0.5, alpha=0.4)

        ax.set_ylabel(f"DoF {dof + 1}")
        ax.grid(True, alpha=0.3)
        if dof == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)

    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Local (Kiemel & Kroeger 2024) vs Cloud (Ruckig Pro)")
    fig.tight_layout()

    out_path = Path(__file__).parent.parent / "build" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved {out_path}")


def _poly_arc_fractions(poly):
    """Return arc-length fractions (0..1) for each vertex of a polyline."""
    diffs = np.diff(poly, axis=0)
    segs = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]
    return arc / total if total > 1e-12 else arc


def plot_path_geometry(filename, ref_poly, runs, resampled, dev_from_ref):
    """Joint-space path comparison figure (2-D only, suitable for joint angles).

    Layout
    ------
    Rows 0..(dofs-1) : one panel per joint – joint angle vs arc-length fraction.
        Each panel shows:
          • reference polyline (dashed black with diamond waypoint markers)
          • local backend (blue)
          • cloud backend (orange)
          • shaded region between each backend and the reference
    Row dofs : aggregate Euclidean deviation from reference + cross-backend diff.
    """
    dofs = ref_poly.shape[1]
    n_rows = dofs + 1          # one joint panel + one deviation panel
    colors   = {"local": "#2196F3", "cloud": "#FF9800"}
    ref_color = "#212121"
    bg_color  = "#F7F7F7"

    wp_labels = (["start"]
                 + [f"wp{i+1}" for i in range(len(ref_poly) - 2)]
                 + ["end"])

    ref_arc_f = _poly_arc_fractions(ref_poly)   # (N_wp,) in [0, 1]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(11, 2.8 * n_rows),
        dpi=120,
        sharex=True,
    )
    fig.patch.set_facecolor("#FAFAFA")

    # ── helper ───────────────────────────────────────────────────────────────
    def _style(ax, ylabel, title):
        ax.set_facecolor(bg_color)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        ax.grid(True, color="white", linewidth=1.0)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#CCCCCC")

    # ── one panel per joint ───────────────────────────────────────────────────
    for d in range(dofs):
        ax = axes[d]

        # reference polyline – draw as step-through-waypoints line
        ax.plot(ref_arc_f, ref_poly[:, d],
                color=ref_color, linewidth=1.8, linestyle="--",
                zorder=5, label="reference (waypoints)")
        ax.scatter(ref_arc_f, ref_poly[:, d],
                   color=ref_color, s=55, zorder=6, marker="D",
                   edgecolors="white", linewidths=0.8)
        # annotate waypoint names on top panel only to avoid clutter
        if d == 0:
            for k, (sx, wy) in enumerate(zip(ref_arc_f, ref_poly[:, d])):
                ax.annotate(wp_labels[k], (sx, wy),
                            textcoords="offset points", xytext=(3, 6),
                            fontsize=6.5, color=ref_color, zorder=7)

        # backend trajectories + deviation shading
        for run_key, pos in resampled.items():
            c = colors.get(run_key, "gray")
            s = np.linspace(0.0, 1.0, len(pos))
            label = runs[run_key][0]

            ax.plot(s, pos[:, d], color=c, linewidth=1.5,
                    alpha=0.9, zorder=4, label=label)

            # deviation fill between this backend and the reference
            ref_interp = np.interp(s, ref_arc_f, ref_poly[:, d])
            ax.fill_between(s, pos[:, d], ref_interp,
                             color=c, alpha=0.14, zorder=2,
                             label=f"{label} deviation")

        _style(ax, f"Joint {d+1} [rad]", f"Joint {d+1}")
        ax.legend(fontsize=7, loc="best", framealpha=0.92,
                  edgecolor="#CCCCCC", fancybox=False, ncol=2)

    # ── deviation panel ───────────────────────────────────────────────────────
    ax_dev = axes[dofs]

    for run_key, dev in dev_from_ref.items():
        c = colors.get(run_key, "gray")
        label = runs[run_key][0]
        s = np.linspace(0.0, 1.0, len(dev))
        ax_dev.fill_between(s, dev, alpha=0.18, color=c)
        ax_dev.plot(s, dev, color=c, linewidth=1.5,
                    label=(f"{label}  "
                           f"max={dev.max():.4f}  "
                           f"mean={dev.mean():.4f}  "
                           f"rms={np.sqrt(np.mean(dev**2)):.4f}"))

    # cross-backend line
    keys = list(runs.keys())
    if len(keys) == 2:
        pos_a, pos_b = resampled[keys[0]], resampled[keys[1]]
        n = min(len(pos_a), len(pos_b))
        cross = np.linalg.norm(pos_a[:n] - pos_b[:n], axis=1)
        s = np.linspace(0.0, 1.0, n)
        ax_dev.fill_between(s, cross, alpha=0.15, color="#4CAF50")
        ax_dev.plot(s, cross, color="#2E7D32", linewidth=1.5, linestyle="-.",
                    label=(f"cross-backend  "
                           f"max={cross.max():.4f}  "
                           f"mean={cross.mean():.4f}  "
                           f"rms={np.sqrt(np.mean(cross**2)):.4f}"))

    _style(ax_dev, "deviation [rad]",
           "Euclidean deviation from piecewise-linear waypoint path")
    ax_dev.set_xlabel("arc-length fraction (time-independent)", fontsize=8)
    ax_dev.legend(fontsize=7.5, loc="upper right", framealpha=0.95,
                  edgecolor="#CCCCCC", fancybox=False)

    fig.suptitle(
        "Joint path comparison: reference waypoints vs local/cloud backends\n"
        "(x-axis: arc-length fraction — time removed)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = Path(__file__).parent.parent / "build" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")

# ---------------------------------------------------------------------------
# Kinematic-validity check
# ---------------------------------------------------------------------------

def check_kinematic_limits(inp, runs, dt=1e-4, tol_factor=1.05):
    """Sample each trajectory at fine resolution and verify that velocity,
    acceleration, and jerk stay within the specified limits (with a small
    tolerance factor to allow for numerical noise).

    Jerk is estimated via finite differences of the acceleration signal.

    Prints a per-backend, per-DoF report and returns True only if every
    backend passes.
    """
    dofs = inp.degrees_of_freedom
    v_max = np.array(inp.max_velocity)
    a_max = np.array(inp.max_acceleration)
    j_max = np.array(inp.max_jerk)

    all_ok = True

    print("\n--- Kinematic-limits validation ---")
    print(f"Tolerance factor: {tol_factor}  (limits multiplied by this value)")
    print(f"Sample dt: {dt}\n")

    for run_key, (label, out_list) in runs.items():
        traj = out_list[0].trajectory
        duration = traj.duration
        n = int(np.ceil(duration / dt)) + 1
        times = np.linspace(0.0, duration, n)

        pos = np.empty((n, dofs))
        vel = np.empty((n, dofs))
        acc = np.empty((n, dofs))
        for i, t in enumerate(times):
            p, v, a = traj.at_time(t)
            pos[i] = p
            vel[i] = v
            acc[i] = a

        # Jerk via finite differences of acceleration
        jerk = np.diff(acc, axis=0) / dt

        # Per-DoF checks
        backend_ok = True
        header = (f"  {'DoF':>3s}  {'|v|_max':>9s} / {'v_lim':>7s}  "
                  f"{'|a|_max':>9s} / {'a_lim':>7s}  "
                  f"{'|j|_max':>9s} / {'j_lim':>7s}  {'status'}")
        print(f"{label}:")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for d in range(dofs):
            v_peak = np.max(np.abs(vel[:, d]))
            a_peak = np.max(np.abs(acc[:, d]))
            j_peak = np.max(np.abs(jerk[:, d]))

            v_ok = v_peak <= v_max[d] * tol_factor
            a_ok = a_peak <= a_max[d] * tol_factor
            j_ok = j_peak <= j_max[d] * tol_factor
            dof_ok = v_ok and a_ok and j_ok

            flags = []
            if not v_ok:
                flags.append("v!")
            if not a_ok:
                flags.append("a!")
            if not j_ok:
                flags.append("j!")
            status = "OK" if dof_ok else "FAIL " + " ".join(flags)

            print(f"  {d+1:3d}  {v_peak:9.4f} / {v_max[d]:7.3f}  "
                  f"{a_peak:9.4f} / {a_max[d]:7.3f}  "
                  f"{j_peak:9.4f} / {j_max[d]:7.3f}  {status}")

            if not dof_ok:
                backend_ok = False

        result = "PASS" if backend_ok else "FAIL"
        print(f"  => {label}: {result}\n")
        if not backend_ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runs = {}

    try:
        otg_local, out_local = run_backend(WaypointsBackend.Local, "Local")
        print(f"local  duration = {out_local[0].trajectory.duration:0.4f} s")
        runs["local"] = ("local", out_local)
    except Exception as e:
        print(f"[skip] local backend: {e}")

    try:
        otg_cloud, out_cloud = run_backend(WaypointsBackend.Cloud, "Cloud")
        print(f"cloud  duration = {out_cloud[0].trajectory.duration:0.4f} s")
        runs["cloud"] = ("cloud", out_cloud)
    except Exception as e:
        print(f"[skip] cloud backend: {e}")

    if len(runs) == 0:
        raise SystemExit("no backends produced a trajectory")

    inp_ref = make_input()
    ref_poly = build_reference_polyline(inp_ref)

    # ---- kinematic-validity check ----
    check_kinematic_limits(inp_ref, runs)

    # ---- deviation analysis ----
    dev_from_ref, resampled = compare_paths(runs, ref_poly)

    # ---- plots ----
    plot_compare("18_backend_compare.png", inp_ref, runs)
    plot_path_geometry("18_path_geometry.png", ref_poly, runs, resampled, dev_from_ref)
