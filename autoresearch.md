# Autoresearch: Local Waypoints Calculator Parity with Cloud API

## Objective
Close the gap between the local waypoints calculator (LocalWaypointsCalculator) and the cloud API (Ruckig Pro) for trajectory duration and path deviation. The local implementation should match the paper's algorithm (Kiemel & Kröger 2024, arXiv:2407.13423) correctly.

Current best:
- local  duration = 7.87s (matching cloud 7.87s, -0.01% gap)
- local max deviation = 0.416 (cloud = 0.260 — still 60% higher)
- Kinematic limits: PASS
- End state: PASS (position/velocity correct at trajectory end)

## Metrics
- **Primary**: duration_gap_pct (%, lower/more-negative is better) — percentage difference between local and cloud duration
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
- End state MUST match target (position, velocity error < 0.05)
- Must not break the existing Ruckig API
- No new dependencies

## What's Been Tried

### Major Wins
1. **True local extrema only** (§III-B) - Removed forced v=0 at monotonic pass-through waypoints. Duration 9.81→8.59s
2. **Remove pre-alignment** - Pass initial v/a directly to core instead of braking to rest first. Duration 8.59→8.20s
3. **Fix forward_position_to_s for braking overshoot** - When initial velocity opposes section direction, position overshoots p_start during braking. The function was mapping these positions to wrong sections. Duration 8.20→7.87s (matching cloud!)
4. **Extended section bounds for braking overshoot** - Upper/lower trajectory validity checks now allow overshoot on the starting side
5. **Added end-state accuracy check to autoresearch.sh** - Prevents accepting trajectories that reach wrong position/velocity

### Dead Ends
- **More tracking iterations (100, 200)** - No change, already converged at 50
- **Smaller delta_u_ref (0.005)** - Worse duration, kinematic failure
- **Larger delta_u_ref (0.02)** - Slightly worse on all metrics
- **Set a=m*af_fast at section transitions** - Kinematic failure, jerk limit violations
- **Smaller sim_dt (0.001)** - No improvement in deviation, slower compute
- **All waypoints as section boundaries** - Duration much worse (9.21→10.43s), deviation barely improves
- **Pass-through velocity at waypoints** - Catastrophic: broke tracking scheme
- **Multi-region update_u_ref** - No improvement when combined with slowest DoF u_ref
- **Average u_ref + multi-region update** - Max_dev 0.331 (better!) but duration 8.95 (+13.76%)
- **Element-wise min of all DoFs** - Duration 9.13% worse, kinematic fail
- **Best iteration by max_dev** - Worse duration (3.64% gap)
- **Pass target v/a directly to core** - WRONG: end state doesn't match target (duration was -9% but position error 0.17, velocity error 0.81)

### Root Cause of Remaining Deviation (0.416 vs cloud 0.260)
- At worst-deviation time (t≈3.85s), u values spread: u_dof1≈4.32, u_dof2≈5.36, u_dof3≈4.51 (spread=1.04)
- Cloud at same proportional time: spread=0.45 (much better)
- DoF 2 is ahead because: starts with v=0.2 in wrong direction → overshoots quickly → tracking can't slow it down enough
- DoF 2 at v≈0.8, sim_dt=0.0025: min progress/step≈0.002 units, rate=0.8 units/s. u_ref advances at ~0.5 units/s. Physically impossible to hold DoF 2 in sync.
- Kinematic constraint: j_max=10 for DoF 2. To stop DoF 2 instantly would require j≈-1920, but j_max=10.
- The paper's algorithm assumes s_target always achievable by choosing m∈[0,1]. This breaks when s_target<s_lower (brake still too fast).
- Post-alignment (target v/a from zero) takes 0.751s and CANNOT be eliminated (target_v opposes last section direction for some DoFs)

### What Remains to Try
1. Larger delta_u_ref (0.05-0.1) to reduce ping-pong in update_u_ref
2. Different initial u_ref from only-extrema DoFs' trajectories (exclude fast DoFs from initial u_ref)
3. Reset tracking when DoF overshoot exceeds a threshold
