# Local Waypoints Devlog

## 2026-04-25

- Started implementing the local waypoint backend from `doc/Jerk-limited_Traversal_of_One-dimensional_Paths_and_its_Application_to_Multi-dimensional_Path_Tracking.pdf`.
- Confirmed `include/ruckig/calculator_local.hpp` is empty scaffolding and `Calculator` already dispatches intermediate-position trajectories to `LocalWaypointsCalculator` when `WITH_LOCAL_WAYPOINTS` is enabled.
- Extracted the relevant paper workflow:
  - compute feasible waypoint accelerations for each one-dimensional path with binary search;
  - construct upper and lower jerk-limited trajectories for bounded progress;
  - use mapping factors to track a shared multi-dimensional path-length reference.
- Implemented an initial local calculator checkpoint:
  - scalar Ruckig calls for point-to-point jerk-limited motion;
  - binary searches for feasible incoming/outgoing waypoint accelerations;
  - section-wise multi-DoF trajectory assembly using the existing `Trajectory` profile storage.
- Verified the checkpoint with the requested Python example. Kinematic limits pass, but duration is still high because every sampled waypoint is treated as a stop; the next implementation slice will collapse each one-dimensional path to its extrema, matching the paper's Section III-B setup more closely.
- Reworked the multi-dimensional path assembly:
  - extract each DoF's one-dimensional path extrema before computing feasible accelerations;
  - choose the slowest one-dimensional traversal as the shared path-length reference `u(t)`;
  - generate bounded short tracking sections at the paper's 2.5 ms step when the control cycle is coarser;
  - pack any Ruckig brake pre-trajectory into those short sections so later trajectory sections remain acceleration-continuous.
- Verified with the requested command. Local backend duration on `examples/18_waypoints_backend_compare.py` is 2.5396 s and the sampled velocity, acceleration, and jerk checks pass.
- Ran `test/local_waypoints_diagnostics.py`; all diagnostic checks pass.
- Fixed waypoint handling for local trajectories with internal tracking slices:
  - added an internal-to-public section mapping in `Trajectory` so `OutputParameter.pass_to_input` only removes original intermediate waypoints, not local backend tracking slices;
  - changed scalar-position to path-length mapping to interpolate through the original waypoint span instead of linearly between compressed extrema endpoints;
  - added a diagnostic that catches public section numbers leaking internal slice indices.
- Verified the user-provided `make_input_04` path in `examples/18_waypoints_backend_compare.py`: local kinematic validation passes, local duration is 8.1286 s, and local path-deviation max/mean are 0.166686 / 0.095980.
