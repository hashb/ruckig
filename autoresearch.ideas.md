# Autoresearch Ideas (for future experiments)

## Deviation Reduction

- **Multi-region update_u_ref**: Current update only adjusts 1 region per iteration (greedy).
  Try adjusting top-K regions per iteration (K=3-5) for faster convergence.
  
- **Simultaneous undershoot+overshoot fix**: When lowering u_ref for an undershoot region,
  also raise it for the corresponding overshoot region in the same iteration.
  
- **TOPP-based u_ref**: Use proper time-optimal path parameterization (minimax velocity/jerk)
  to compute u_ref directly instead of the greedy slow-DoF approach. Complex to implement.
  
- **Waypoint-aware u_ref**: Build u_ref as piecewise linear through waypoint timing checkpoints,
  where each checkpoint time = max(t_all_DoFs_reach_waypoint). This ensures no DoF is "behind"
  when u_ref passes through a waypoint's u value. Might increase duration slightly.
  
- **Better s_target clamping**: When the brake trajectory can't slow a DoF enough
  (s_lower > s_target), detect this and adjust u_ref immediately in that iteration
  rather than waiting for the next update_u_ref call.
  
- **Phase 2 post-processing**: After finding the best u_ref from 50 iterations, run
  a second optimization phase specifically targeting high-deviation regions.
