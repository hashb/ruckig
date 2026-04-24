# Local Waypoints Devlog

## 2026-04-25

- Started implementing the local waypoint backend from `doc/Jerk-limited_Traversal_of_One-dimensional_Paths_and_its_Application_to_Multi-dimensional_Path_Tracking.pdf`.
- Confirmed `include/ruckig/calculator_local.hpp` is empty scaffolding and `Calculator` already dispatches intermediate-position trajectories to `LocalWaypointsCalculator` when `WITH_LOCAL_WAYPOINTS` is enabled.
- Extracted the relevant paper workflow:
  - compute feasible waypoint accelerations for each one-dimensional path with binary search;
  - construct upper and lower jerk-limited trajectories for bounded progress;
  - use mapping factors to track a shared multi-dimensional path-length reference.
