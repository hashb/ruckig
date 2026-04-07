# Compare the local (Kiemel & Kroeger 2024) and cloud (Ruckig Pro) waypoints
# backends side-by-side, plotting both trajectories on the same axes.
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
    inp.current_position = [0.0, 0.0, 0.0]
    inp.current_velocity = [0.0, 0.0, 0.0]
    inp.current_acceleration = [0.0, 0.0, 0.0]

    inp.intermediate_positions = [
        [1.0, 0.5, -0.2],
        [0.4, 1.5, 0.6],
        [-0.3, 0.2, 1.0],
    ]

    inp.target_position = [0.5, 1.0, 0.0]
    inp.target_velocity = [0.0, 0.0, 0.0]
    inp.target_acceleration = [0.0, 0.0, 0.0]

    inp.max_velocity = [1.0, 1.0, 1.0]
    inp.max_acceleration = [2.0, 2.0, 2.0]
    inp.max_jerk = [6.0, 6.0, 6.0]
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

    plot_compare("18_backend_compare.pdf", make_input(), runs)
