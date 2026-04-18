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

### Major Wins
1. **True local extrema only** (§III-B) - Removed forced v=0 at monotonic pass-through waypoints. Duration 9.81→8.59s
2. **Remove pre-alignment** - Pass initial v/a directly to core instead of braking to rest first. Duration 8.59→8.20s
3. **Fix forward_position_to_s for braking overshoot** - When initial velocity opposes section direction, the position overshoots p_start during braking. The forward_position_to_s function was mapping these positions to wrong sections, corrupting s_dt and m calculations. Duration 8.20→7.87s (matching cloud!)
4. **Extended section bounds for braking overshoot** - Upper/lower trajectory validity checks now allow overshoot on the starting side

### Dead Ends
- **More tracking iterations (100)** - No change, already converged at 50
- **Smaller delta_u_ref (0.005)** - Worse duration, kinematic failure
- **Larger delta_u_ref (0.02)** - Slightly worse on all metrics
- **Set a=m*af_fast at section transitions** - Kinematic failure, jerk limit violations
- **Smaller sim_dt (0.001)** - No improvement in deviation, slower compute
- **All waypoints as section boundaries** - Duration much worse (9.21→10.43s), deviation barely improves
- **Pass-through velocity at waypoints** - Catastrophic: broke tracking scheme

### Root Cause of Remaining Deviation (0.42 vs cloud 0.26)
The deviation is caused by DoFs being at DIFFERENT u values near waypoints. Each DoF passes through pass-through waypoints accurately, but at different times. At any given time, DoFs are at different progress levels, causing the combined position to deviate from the reference polyline.

The cloud API avoids this by stopping at every waypoint (v=0), naturally synchronizing all DoFs. The paper's algorithm only enforces synchronization at true local extrema.

### Experiment Ideas (priority order)
1. **Improve u_ref initialization** - Include waypoint timing constraints so u_ref aligns all DoFs at each waypoint
2. **Waypoint-enforced u_ref checkpoints** - After initial tracking, add u_ref corrections at each waypoint
3. **Modified update_u_ref** - Use path deviation (distance to polyline) instead of position deviation
4. **Two-phase tracking** - First phase matches duration, second phase optimizes deviation
