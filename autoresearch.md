# Autoresearch: Local Waypoints Calculator Parity with Cloud API

## Objective
Close the gap between the local waypoints calculator (LocalWaypointsCalculator) and the cloud API (Ruckig Pro) for trajectory duration and path deviation. The local implementation should match the paper's algorithm (Kiemel & Kröger 2024, arXiv:2407.13423) correctly.

Current state:
- local  duration = 9.8108 s
- cloud  duration = 7.8705 s  (25% faster)
- local max deviation = 0.265393
- cloud max deviation = 0.260108

## Metrics
- **Primary**: duration_gap_pct (%, lower is better) — percentage difference between local and cloud duration
- **Secondary**: local_duration, cloud_duration, local_max_dev, cloud_max_dev, kinematic_pass (bool)

## How to Run
`./autoresearch.sh` — outputs `METRIC` lines with duration and deviation data.

## Files in Scope
- `include/ruckig/calculator_local.hpp` — The main file. Contains LocalWaypointsCalculator implementing the 3-stage paper algorithm (§III-B/C/D). All optimization work happens here.
- `examples/18_waypoints_backend_compare.py` — Test/benchmark script. Read-only.

## Off Limits
- `include/ruckig/calculator_target.hpp` — Target calculator (used as backend, not to be modified)
- `include/ruckig/calculator_cloud.hpp` — Cloud calculator
- `src/` — C++ source files
- `CMakeLists.txt`, `pyproject.toml`, `Cargo.toml` — Build system files
- Any file not listed in "Files in Scope"

## Constraints
- Kinematic limits MUST remain valid (velocity, acceleration, jerk within bounds)
- Must not break the existing Ruckig API
- No new dependencies

## What's Been Tried
(Updated as experiments accumulate)

### Root Cause Analysis
The paper says per-DoF sections should be between TRUE LOCAL EXTREMA only (where direction changes). The current `build_geometry()` includes EVERY multi-dim waypoint as a section boundary, forcing v=0 at every waypoint for every DoF. This is overly conservative and causes the ~25% duration gap.

The paper's §III-B explicitly states: "The intermediate waypoints pI and pII are local extrema of the path. Consequently, their corresponding velocities also need to be zero." — only for local extrema.

The paper's §III-D tracking scheme ensures all DoFs approximately pass through every waypoint via the u_ref parameter, even without forcing v=0.

### Experiment Ideas (priority order)
1. **Switch build_geometry to true local extrema only** — The biggest win. Remove the code that includes every waypoint; only include waypoints where the per-DoF direction actually changes (or at start/end).
2. **Improve online tracking for monotonic sections** — When a DoF passes through a waypoint without stopping, ensure the u_ref tracking keeps it close to the waypoint position.
3. **Refine binary search for accel ranges** — The current binary search is correct per the paper; likely not a major source of duration difference.
4. **Adjust sim_dt and tracking parameters** — May need finer discretization for smooth section transitions.
5. **Handle section transitions with non-zero velocity** — After switching to true extrema, sections that span multiple waypoints will need to handle non-zero velocity at intermediate waypoints correctly.
