#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include <ruckig/calculator_target.hpp>
#include <ruckig/error.hpp>
#include <ruckig/input_parameter.hpp>
#include <ruckig/profile.hpp>
#include <ruckig/result.hpp>
#include <ruckig/trajectory.hpp>


namespace ruckig {

//! @brief Offline calculation class for trajectories with intermediate waypoints.
//!
//! Implements the algorithm of:
//!     J. C. Kiemel and T. Kröger,
//!     "Jerk-limited Traversal of One-dimensional Paths and its Application to
//!      Multi-dimensional Path Tracking", arXiv:2407.13423, 2024.
//!
//! Each intermediate waypoint is treated as a path waypoint with zero velocity.
//! For every waypoint a per-DoF binary search determines the largest feasible
//! target acceleration that does not cause a position overshoot. Each section
//! between two waypoints is then computed individually with the existing
//! state-to-state TargetCalculator and the resulting profiles are concatenated
//! into the output trajectory.
//!
//! Compared to the cloud-based Ruckig Pro implementation this calculator is
//! offline (no network round trip) and self-contained, but it assumes the
//! intermediate waypoints describe a path with v=0 at every waypoint. For
//! arbitrary intermediate states it is therefore generally not time-optimal.
template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
class LocalWaypointsCalculator {
    template<class T> using Vector = CustomVector<T, DOFs>;

    //! Per-segment state-to-state calculator (uses the existing solver).
    TargetCalculator<DOFs, CustomVector> segment_calc;
    InputParameter<DOFs, CustomVector> segment_input;
    Trajectory<DOFs, CustomVector> segment_traj;

    //! 1-DoF state-to-state calculator used by the binary search.
    TargetCalculator<0, StandardVector> bs_calc;
    InputParameter<0, StandardVector> bs_input;
    Trajectory<0, StandardVector> bs_traj;

    //! Storage for the per-waypoint, per-DoF acceleration values.
    std::vector<std::vector<double>> waypoint_accelerations;

    //! Number of binary search refinement steps after the upper bound is rejected.
    static constexpr int binary_search_iterations {16};
    //! Tolerance used when checking for position overshoot.
    static constexpr double overshoot_tolerance {1e-9};
    //! Threshold below which two waypoint coordinates are treated as identical.
    static constexpr double position_eps {1e-12};

    //! Initialise the 1-DoF binary-search input with empty min limits placeholders.
    void initialize_binary_search_input() {
        bs_input.min_velocity = std::vector<double>(1, 0.0);
        bs_input.min_acceleration = std::vector<double>(1, 0.0);
    }

    //! Test whether a 1-DoF transition (p0,v0,a0) -> (pf,vf,af) is valid.
    //!
    //! "Valid" means that Ruckig finds a state-to-state trajectory that
    //! respects all kinematic limits AND that the position never overshoots
    //! the bounding box [min(p0,pf), max(p0,pf)]. The latter is the criterion
    //! used by the paper to detect a velocity sign change (since v0=vf=0).
    bool test_segment(double p0, double v0, double a0,
                      double pf, double vf, double af,
                      double vmax, double vmin,
                      double amax, double amin,
                      double jmax) {
        bs_input.current_position[0] = p0;
        bs_input.current_velocity[0] = v0;
        bs_input.current_acceleration[0] = a0;
        bs_input.target_position[0] = pf;
        bs_input.target_velocity[0] = vf;
        bs_input.target_acceleration[0] = af;
        bs_input.max_velocity[0] = vmax;
        bs_input.max_acceleration[0] = amax;
        bs_input.max_jerk[0] = jmax;
        (*bs_input.min_velocity)[0] = vmin;
        (*bs_input.min_acceleration)[0] = amin;
        bs_input.enabled[0] = true;
        bs_input.control_interface = ControlInterface::Position;
        bs_input.synchronization = Synchronization::Time;
        bs_input.duration_discretization = DurationDiscretization::Continuous;
        bs_input.minimum_duration = std::nullopt;

        bool seg_interrupted = false;
        const Result r = bs_calc.template calculate<false>(bs_input, bs_traj, 0.0, seg_interrupted);
        if (r != Result::Working) {
            return false;
        }

        const Profile& prof = bs_traj.profiles[0][0];
        const Bound extrema = prof.get_position_extrema();
        const double p_low = std::min(p0, pf);
        const double p_high = std::max(p0, pf);
        return extrema.min >= p_low - overshoot_tolerance
            && extrema.max <= p_high + overshoot_tolerance;
    }

    //! Binary search for the maximum feasible target acceleration when leaving
    //! a waypoint, i.e. find the largest a_out >= 0 such that the transition
    //! (p_prev, 0, 0) -> (p_curr, 0, sign*a_out) is valid.
    double find_max_a_out(double p_prev, double p_curr, double sign,
                          double vmax, double vmin,
                          double amax, double amin, double jmax) {
        if (std::abs(p_curr - p_prev) < position_eps || amax <= 0.0) {
            return 0.0;
        }
        if (test_segment(p_prev, 0.0, 0.0, p_curr, 0.0, sign * amax,
                         vmax, vmin, amax, amin, jmax)) {
            return amax;
        }
        double low = 0.0;
        double high = amax;
        for (int i = 0; i < binary_search_iterations; ++i) {
            const double mid = 0.5 * (low + high);
            if (test_segment(p_prev, 0.0, 0.0, p_curr, 0.0, sign * mid,
                             vmax, vmin, amax, amin, jmax)) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return low;
    }

    //! Binary search for the maximum feasible input acceleration when entering
    //! the next section: find the largest a_in >= 0 such that the transition
    //! (p_curr, 0, sign*a_in) -> (p_next, 0, 0) is valid.
    double find_max_a_in(double p_curr, double p_next, double sign,
                         double vmax, double vmin,
                         double amax, double amin, double jmax) {
        if (std::abs(p_next - p_curr) < position_eps || amax <= 0.0) {
            return 0.0;
        }
        if (test_segment(p_curr, 0.0, sign * amax, p_next, 0.0, 0.0,
                         vmax, vmin, amax, amin, jmax)) {
            return amax;
        }
        double low = 0.0;
        double high = amax;
        for (int i = 0; i < binary_search_iterations; ++i) {
            const double mid = 0.5 * (low + high);
            if (test_segment(p_curr, 0.0, sign * mid, p_next, 0.0, 0.0,
                             vmax, vmin, amax, amin, jmax)) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return low;
    }

    //! Look up the position vector of the i-th waypoint.
    const Vector<double>& waypoint_at(const InputParameter<DOFs, CustomVector>& input,
                                       size_t i, size_t n_waypoints) const {
        if (i == 0) {
            return input.current_position;
        }
        if (i == n_waypoints - 1) {
            return input.target_position;
        }
        return input.intermediate_positions[i - 1];
    }

public:
    size_t degrees_of_freedom;

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator():
        bs_calc(TargetCalculator<0, StandardVector>(1)),
        bs_input(InputParameter<0, StandardVector>(1)),
        bs_traj(Trajectory<0, StandardVector>(1)),
        degrees_of_freedom(DOFs)
    {
        initialize_binary_search_input();
    }

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t /*max_number_of_waypoints*/):
        LocalWaypointsCalculator()
    {
    }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs):
        segment_calc(TargetCalculator<DOFs, CustomVector>(dofs)),
        segment_input(InputParameter<DOFs, CustomVector>(dofs)),
        segment_traj(Trajectory<DOFs, CustomVector>(dofs)),
        bs_calc(TargetCalculator<0, StandardVector>(1)),
        bs_input(InputParameter<0, StandardVector>(1)),
        bs_traj(Trajectory<0, StandardVector>(1)),
        degrees_of_freedom(dofs)
    {
        initialize_binary_search_input();
    }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs, size_t /*max_number_of_waypoints*/):
        LocalWaypointsCalculator(dofs)
    {
    }

    //! Calculate a trajectory that visits the input's intermediate positions in order.
    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input,
                     Trajectory<DOFs, CustomVector>& traj,
                     double delta_time, bool& was_interrupted) {
        was_interrupted = false;

        const size_t n_intermediates = input.intermediate_positions.size();
        const size_t n_sections = n_intermediates + 1;
        const size_t n_waypoints = n_intermediates + 2;

        traj.degrees_of_freedom = degrees_of_freedom;
        traj.resize(n_intermediates);
        traj.continue_calculation_counter = 0;

        // Allocate per-waypoint acceleration storage.
        if (waypoint_accelerations.size() < n_waypoints) {
            waypoint_accelerations.resize(n_waypoints);
        }
        for (size_t i = 0; i < n_waypoints; ++i) {
            waypoint_accelerations[i].assign(degrees_of_freedom, 0.0);
        }
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            waypoint_accelerations[0][dof] = input.current_acceleration[dof];
            waypoint_accelerations[n_waypoints - 1][dof] = input.target_acceleration[dof];
        }

        // Find the largest feasible acceleration at every intermediate waypoint
        // by running the binary search of Section III-B of the paper independently
        // for every DoF.
        for (size_t i = 1; i < n_waypoints - 1; ++i) {
            const Vector<double>& p_prev = waypoint_at(input, i - 1, n_waypoints);
            const Vector<double>& p_curr = waypoint_at(input, i, n_waypoints);
            const Vector<double>& p_next = waypoint_at(input, i + 1, n_waypoints);

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                if (!input.enabled[dof]) {
                    waypoint_accelerations[i][dof] = 0.0;
                    continue;
                }

                const double dir_in = p_curr[dof] - p_prev[dof];
                const double dir_out = p_next[dof] - p_curr[dof];

                // Constant or aligned direction in this DoF: not a true extremum
                // of the 1-D path, fall back to the safe choice a=0.
                if (std::abs(dir_in) < position_eps || std::abs(dir_out) < position_eps) {
                    waypoint_accelerations[i][dof] = 0.0;
                    continue;
                }
                if (dir_in * dir_out > 0.0) {
                    waypoint_accelerations[i][dof] = 0.0;
                    continue;
                }

                // Local minimum (dir_in<0, dir_out>0) -> a >= 0
                // Local maximum (dir_in>0, dir_out<0) -> a <= 0
                const double sign = (dir_out > 0.0) ? 1.0 : -1.0;

                const double vmax = input.max_velocity[dof];
                const double amax = input.max_acceleration[dof];
                const double jmax = input.max_jerk[dof];
                const double vmin = input.min_velocity ? (*input.min_velocity)[dof] : -vmax;
                const double amin = input.min_acceleration ? (*input.min_acceleration)[dof] : -amax;

                const double aout_max = find_max_a_out(p_prev[dof], p_curr[dof], sign,
                                                        vmax, vmin, amax, amin, jmax);
                const double ain_max = find_max_a_in(p_curr[dof], p_next[dof], sign,
                                                      vmax, vmin, amax, amin, jmax);

                waypoint_accelerations[i][dof] = sign * std::min(aout_max, ain_max);
            }
        }

        // Compute every section as an independent state-to-state problem and
        // append it to the output trajectory.
        double cumulative = 0.0;
        for (size_t s = 0; s < n_sections; ++s) {
            const Vector<double>& p_start = waypoint_at(input, s, n_waypoints);
            const Vector<double>& p_end = waypoint_at(input, s + 1, n_waypoints);

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                segment_input.current_position[dof] = p_start[dof];
                segment_input.target_position[dof] = p_end[dof];

                segment_input.current_velocity[dof] = (s == 0)
                    ? input.current_velocity[dof] : 0.0;
                segment_input.target_velocity[dof] = (s == n_sections - 1)
                    ? input.target_velocity[dof] : 0.0;

                segment_input.current_acceleration[dof] = waypoint_accelerations[s][dof];
                segment_input.target_acceleration[dof] = waypoint_accelerations[s + 1][dof];

                segment_input.max_velocity[dof] = input.max_velocity[dof];
                segment_input.max_acceleration[dof] = input.max_acceleration[dof];
                segment_input.max_jerk[dof] = input.max_jerk[dof];
                segment_input.enabled[dof] = input.enabled[dof];
            }

            segment_input.min_velocity = input.min_velocity;
            segment_input.min_acceleration = input.min_acceleration;
            segment_input.control_interface = ControlInterface::Position;
            segment_input.synchronization = (input.synchronization == Synchronization::None)
                ? Synchronization::Time
                : input.synchronization;
            segment_input.duration_discretization = DurationDiscretization::Continuous;
            segment_input.per_dof_control_interface = std::nullopt;
            segment_input.per_dof_synchronization = std::nullopt;

            if (input.per_section_minimum_duration
                && s < input.per_section_minimum_duration->size()) {
                segment_input.minimum_duration = (*input.per_section_minimum_duration)[s];
            } else {
                segment_input.minimum_duration = std::nullopt;
            }

            bool seg_interrupted = false;
            const Result r = segment_calc.template calculate<throw_error>(
                segment_input, segment_traj, delta_time, seg_interrupted);
            if (r != Result::Working) {
                return r;
            }

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                traj.profiles[s][dof] = segment_traj.profiles[0][dof];
            }
            cumulative += segment_traj.duration;
            traj.cumulative_times[s] = cumulative;
        }

        traj.duration = cumulative;
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            traj.independent_min_durations[dof] = cumulative;
        }

        return Result::Working;
    }

    //! continue_calculation is not supported by the local backend.
    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>&,
                                 Trajectory<DOFs, CustomVector>&,
                                 double, bool&) {
        if constexpr (throw_error) {
            throw RuckigError("continue calculation not available in local waypoints backend.");
        }
        return Result::Error;
    }
};

} // namespace ruckig
