#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
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
//! Implements the full algorithm of:
//!     J. C. Kiemel and T. Kröger,
//!     "Jerk-limited Traversal of One-dimensional Paths and its Application to
//!      Multi-dimensional Path Tracking", arXiv:2407.13423, 2024.
//!
//! Stages:
//!   1. (III-B) Per-DoF binary search for max feasible acceleration at each
//!      local extremum waypoint.
//!   2. (III-C) 1-D traversal: for each per-DoF section (between extrema),
//!      compute upper (fast) and lower (slow) trajectories. Use mapping
//!      factor m (Eq. 5) to interpolate progress.
//!   3. (III-D) Multi-dim iterative tracking: slowest dimension defines u_ref.
//!      Other dimensions track u_ref by choosing m at each dt. u_ref is
//!      iteratively refined in worst-deviation regions.
template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
class LocalWaypointsCalculator {
    template<class T> using Vector = CustomVector<T, DOFs>;

    TargetCalculator<DOFs, CustomVector> segment_calc;
    InputParameter<DOFs, CustomVector> segment_input;
    Trajectory<DOFs, CustomVector> segment_traj;

    TargetCalculator<0, StandardVector> bs_calc;
    InputParameter<0, StandardVector> bs_input;
    Trajectory<0, StandardVector> bs_traj;
    Trajectory<0, StandardVector> bs_traj_upper;

    static constexpr int binary_search_iterations {16};
    static constexpr double overshoot_tolerance {1e-9};
    static constexpr double position_eps {1e-12};
    static constexpr int n_tracking_iterations {20};
    static constexpr double delta_u_ref {0.01};

    double sim_dt {0.0025};

    struct DofLimits {
        double vmax, vmin, amax, amin, jmax;
    };

    //! Per-DoF 1-D path: only sections between local extrema.
    struct DofPath {
        std::vector<size_t> extrema_wp;        // multi-dim waypoint indices
        std::vector<double> extrema_pos;       // position at each extremum
        std::vector<double> seg_lengths;       // path length per 1-D section
        std::vector<double> cum_s;             // cumulative path length
        std::vector<double> accel_fast;        // signed a_fast at each extremum
        double total_s {0.0};
    };

    //! Sampled 1-D trajectory for a single section.
    struct SectionTrajectory {
        double duration {0.0};
        // Sampled state at each dt step (including initial state).
        std::vector<double> times;
        std::vector<double> positions;
        std::vector<double> velocities;
        std::vector<double> accelerations;
    };

    //! Per-DoF data during simulation.
    struct DofSimState {
        size_t current_section {0};   // index into DofPath sections
        double section_elapsed {0.0}; // time spent in current section
        size_t traj_idx {0};          // index of current SectionTrajectory
    };

    std::vector<DofPath> dof_paths;
    std::vector<DofLimits> dof_limits;
    std::vector<double> global_wp_pos;  // [wp * n_dof + dof]
    std::vector<double> cum_u;
    double total_u {0.0};
    size_t n_waypoints_ {0};

    void initialize_binary_search_input() {
        bs_input.min_velocity = std::vector<double>(1, 0.0);
        bs_input.min_acceleration = std::vector<double>(1, 0.0);
    }

    bool solve_1dof(double p0, double v0, double a0,
                    double pf, double vf, double af,
                    const DofLimits& lim,
                    std::optional<double> min_duration = std::nullopt) {
        bs_input.current_position[0] = p0;
        bs_input.current_velocity[0] = v0;
        bs_input.current_acceleration[0] = a0;
        bs_input.target_position[0] = pf;
        bs_input.target_velocity[0] = vf;
        bs_input.target_acceleration[0] = af;
        bs_input.max_velocity[0] = lim.vmax;
        bs_input.max_acceleration[0] = lim.amax;
        bs_input.max_jerk[0] = lim.jmax;
        (*bs_input.min_velocity)[0] = lim.vmin;
        (*bs_input.min_acceleration)[0] = lim.amin;
        bs_input.enabled[0] = true;
        bs_input.control_interface = ControlInterface::Position;
        bs_input.synchronization = Synchronization::Time;
        bs_input.duration_discretization = DurationDiscretization::Continuous;
        bs_input.minimum_duration = min_duration;
        bool interrupted = false;
        return bs_calc.template calculate<false>(bs_input, bs_traj, 0.0, interrupted)
               == Result::Working;
    }

    //! Braking trajectory in velocity control mode: bring the DoF to
    //! (v=0, a=0) as quickly as the jerk limit allows, regardless of
    //! position. This is the paper's "lower trajectory" at the start of
    //! a section (§III-C).
    bool solve_brake(double p0, double v0, double a0,
                     const DofLimits& lim) {
        bs_input.current_position[0] = p0;
        bs_input.current_velocity[0] = v0;
        bs_input.current_acceleration[0] = a0;
        bs_input.target_position[0] = p0;
        bs_input.target_velocity[0] = 0.0;
        bs_input.target_acceleration[0] = 0.0;
        bs_input.max_velocity[0] = lim.vmax;
        bs_input.max_acceleration[0] = lim.amax;
        bs_input.max_jerk[0] = lim.jmax;
        (*bs_input.min_velocity)[0] = lim.vmin;
        (*bs_input.min_acceleration)[0] = lim.amin;
        bs_input.enabled[0] = true;
        bs_input.control_interface = ControlInterface::Velocity;
        bs_input.synchronization = Synchronization::Time;
        bs_input.duration_discretization = DurationDiscretization::Continuous;
        bs_input.minimum_duration = std::nullopt;
        bool interrupted = false;
        bool ok = bs_calc.template calculate<false>(bs_input, bs_traj, 0.0, interrupted)
                  == Result::Working;
        bs_input.control_interface = ControlInterface::Position;
        return ok;
    }

    void sample_traj(const Trajectory<0, StandardVector>& t, double time,
                     double& p, double& v, double& a) {
        std::vector<double> pp(1), vv(1), aa(1);
        t.at_time(time, pp, vv, aa);
        p = pp[0]; v = vv[0]; a = aa[0];
    }

    //! Check that a solved 1-DoF trajectory stays within the position bounds
    //! [min(p0,pf), max(p0,pf)] AND (when check_velocity_sign is true) that
    //! velocity never reverses sign relative to the section direction.
    //!
    //! The velocity-sign check implements the paper §III-B validity criterion
    //! and should be enabled when both endpoints have v=0 (the binary-search
    //! context). When the trajectory starts or ends with non-zero velocity,
    //! a velocity sign reversal may be physically required to brake and then
    //! proceed, so only position bounds are checked.
    bool trajectory_stays_on_section(const Trajectory<0, StandardVector>& traj,
                                     double p0, double pf,
                                     bool check_velocity_sign = true) const {
        const Profile& prof = traj.profiles[0][0];
        const Bound extrema = prof.get_position_extrema();
        const double p_low = std::min(p0, pf);
        const double p_high = std::max(p0, pf);
        if (extrema.min < p_low - overshoot_tolerance
            || extrema.max > p_high + overshoot_tolerance) {
            return false;
        }

        if (!check_velocity_sign) return true;

        // Check velocity never reverses sign against the section direction.
        const double sign = (pf >= p0) ? 1.0 : -1.0;
        const double T = traj.get_duration();
        if (T < 1e-15) return true;

        const double dense_dt = std::max(T / 100.0, 1e-6);
        for (double t = 0.0; t <= T + 1e-12; t += dense_dt) {
            std::vector<double> pv(1), vv(1), av(1);
            traj.at_time(std::min(t, T), pv, vv, av);
            if (sign > 0.0 && vv[0] < -overshoot_tolerance) return false;
            if (sign < 0.0 && vv[0] > overshoot_tolerance) return false;
        }
        return true;
    }

    bool test_segment(double p0, double v0, double a0,
                      double pf, double vf, double af,
                      const DofLimits& lim) {
        if (!solve_1dof(p0, v0, a0, pf, vf, af, lim)) return false;
        return trajectory_stays_on_section(bs_traj, p0, pf);
    }

    double find_max_a(double p_from, double p_to, double sign, bool is_target,
                      const DofLimits& lim) {
        if (std::abs(p_to - p_from) < position_eps) return 0.0;
        const double a_bound = (sign > 0.0) ? lim.amax : std::abs(lim.amin);
        if (a_bound <= 0.0) return 0.0;
        auto try_a = [&](double mag) -> bool {
            return is_target
                ? test_segment(p_from, 0.0, 0.0, p_to, 0.0, sign * mag, lim)
                : test_segment(p_from, 0.0, sign * mag, p_to, 0.0, 0.0, lim);
        };
        if (try_a(a_bound)) return a_bound;
        double lo = 0.0, hi = a_bound;
        for (int i = 0; i < binary_search_iterations; ++i) {
            double mid = 0.5 * (lo + hi);
            if (try_a(mid)) lo = mid; else hi = mid;
        }
        return lo;
    }

    const Vector<double>& waypoint_at(const InputParameter<DOFs, CustomVector>& input,
                                       size_t i) const {
        if (i == 0) return input.current_position;
        if (i == n_waypoints_ - 1) return input.target_position;
        return input.intermediate_positions[i - 1];
    }

    double wp_pos(size_t d, size_t i) const {
        return global_wp_pos[i * degrees_of_freedom + d];
    }

    void build_geometry(const InputParameter<DOFs, CustomVector>& input) {
        global_wp_pos.resize(n_waypoints_ * degrees_of_freedom);
        for (size_t i = 0; i < n_waypoints_; ++i)
            for (size_t d = 0; d < degrees_of_freedom; ++d)
                global_wp_pos[i * degrees_of_freedom + d] = waypoint_at(input, i)[d];

        dof_limits.resize(degrees_of_freedom);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            dof_limits[d].vmax = input.max_velocity[d];
            dof_limits[d].amax = input.max_acceleration[d];
            dof_limits[d].jmax = input.max_jerk[d];
            dof_limits[d].vmin = input.min_velocity ? (*input.min_velocity)[d] : -input.max_velocity[d];
            dof_limits[d].amin = input.min_acceleration ? (*input.min_acceleration)[d] : -input.max_acceleration[d];
        }

        const size_t n_sec = n_waypoints_ - 1;
        cum_u.resize(n_waypoints_);
        cum_u[0] = 0.0;
        total_u = 0.0;
        for (size_t k = 0; k < n_sec; ++k) {
            double sq = 0.0;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double diff = wp_pos(d, k+1) - wp_pos(d, k);
                sq += diff * diff;
            }
            total_u += std::sqrt(sq);
            cum_u[k+1] = total_u;
        }

        dof_paths.resize(degrees_of_freedom);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            auto& dp = dof_paths[d];
            dp.extrema_wp.clear();
            dp.extrema_pos.clear();
            dp.extrema_wp.push_back(0);
            dp.extrema_pos.push_back(wp_pos(d, 0));

            for (size_t i = 1; i < n_waypoints_ - 1; ++i) {
                double dir_in = wp_pos(d, i) - wp_pos(d, i-1);
                double dir_out = wp_pos(d, i+1) - wp_pos(d, i);
                if (std::abs(dir_in) < position_eps || std::abs(dir_out) < position_eps
                    || dir_in * dir_out < 0.0) {
                    dp.extrema_wp.push_back(i);
                    dp.extrema_pos.push_back(wp_pos(d, i));
                }
            }

            dp.extrema_wp.push_back(n_waypoints_ - 1);
            dp.extrema_pos.push_back(wp_pos(d, n_waypoints_ - 1));

            size_t n_ext = dp.extrema_pos.size();
            dp.seg_lengths.resize(n_ext - 1);
            dp.cum_s.resize(n_ext);
            dp.cum_s[0] = 0.0;
            dp.total_s = 0.0;
            for (size_t k = 0; k + 1 < n_ext; ++k) {
                double len = 0.0;
                size_t from = dp.extrema_wp[k];
                size_t to = dp.extrema_wp[k+1];
                for (size_t j = from; j < to; ++j)
                    len += std::abs(wp_pos(d, j+1) - wp_pos(d, j));
                dp.seg_lengths[k] = len;
                dp.total_s += len;
                dp.cum_s[k+1] = dp.total_s;
            }
        }
    }

    //! During online tracking (§III-C), a pure braking trajectory to target
    //! acceleration 0 can become invalid as the current state approaches the
    //! end of a section. In that case, the paper selects the smallest feasible
    //! positive target acceleration a_min by binary search.
    double find_min_online_target_a(double p_from, double v_from, double a_from,
                                    double p_to, double sign,
                                    const DofLimits& lim, double a_max_mag) {
        if (std::abs(p_to - p_from) < position_eps) return 0.0;
        if (a_max_mag <= 0.0) return 0.0;

        auto try_a = [&](double mag) -> bool {
            return test_segment(p_from, v_from, a_from, p_to, 0.0, sign * mag, lim);
        };

        if (try_a(0.0)) return 0.0;
        if (!try_a(a_max_mag)) return a_max_mag;

        double lo = 0.0;
        double hi = a_max_mag;
        for (int i = 0; i < binary_search_iterations; ++i) {
            const double mid = 0.5 * (lo + hi);
            if (try_a(mid)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        return hi;
    }

    void compute_accel_ranges(const InputParameter<DOFs, CustomVector>& input) {
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            auto& dp = dof_paths[d];
            dp.accel_fast.assign(dp.extrema_pos.size(), 0.0);
            if (!input.enabled[d]) continue;

            for (size_t k = 1; k + 1 < dp.extrema_pos.size(); ++k) {
                double p_prev = dp.extrema_pos[k-1];
                double p_curr = dp.extrema_pos[k];
                double p_next = dp.extrema_pos[k+1];
                double dir_in = p_curr - p_prev;
                double dir_out = p_next - p_curr;
                if (std::abs(dir_in) < position_eps || std::abs(dir_out) < position_eps
                    || dir_in * dir_out > 0.0) continue;

                double sign = (dir_out > 0.0) ? 1.0 : -1.0;
                double aout = find_max_a(p_prev, p_curr, sign, true, dof_limits[d]);
                double ain = find_max_a(p_curr, p_next, sign, false, dof_limits[d]);
                dp.accel_fast[k] = sign * std::min(aout, ain);
            }
        }
    }

    //! Solve a complete 1-D section and sample it at dt intervals.
    //! The section goes from (p0, v0, a0) to (pf, vf, af).
    SectionTrajectory solve_and_sample_section(double p0, double v0, double a0,
                                                double pf, double vf, double af,
                                                const DofLimits& lim) {
        SectionTrajectory st;
        if (!solve_1dof(p0, v0, a0, pf, vf, af, lim)) {
            // Fallback: zero target v/accel
            if (!solve_1dof(p0, v0, a0, pf, 0.0, 0.0, lim)) {
                st.duration = 0.0;
                st.times = {0.0};
                st.positions = {p0};
                st.velocities = {v0};
                st.accelerations = {a0};
                return st;
            }
        }
        st.duration = bs_traj.get_duration();

        // Sample at dt intervals
        double t = 0.0;
        while (t < st.duration) {
            double p, v, a;
            sample_traj(bs_traj, t, p, v, a);
            st.times.push_back(t);
            st.positions.push_back(p);
            st.velocities.push_back(v);
            st.accelerations.push_back(a);
            t += sim_dt;
        }
        // Final sample at exact end
        double p, v, a;
        sample_traj(bs_traj, st.duration, p, v, a);
        st.times.push_back(st.duration);
        st.positions.push_back(p);
        st.velocities.push_back(v);
        st.accelerations.push_back(a);

        return st;
    }

    //! Compute all section trajectories for a DoF (fast or slow).
    void compute_section_trajectories(size_t d,
                                       const InputParameter<DOFs, CustomVector>& input,
                                       bool use_fast,
                                       std::vector<SectionTrajectory>& trajs) {
        const auto& dp = dof_paths[d];
        const size_t n_sec = dp.seg_lengths.size();
        trajs.resize(n_sec);

        double v0 = input.current_velocity[d];
        double a0 = input.current_acceleration[d];

        for (size_t s = 0; s < n_sec; ++s) {
            bool is_last = (s + 1 == n_sec);
            double p0 = dp.extrema_pos[s];
            double pf = dp.extrema_pos[s+1];
            double vf = is_last ? input.target_velocity[d] : 0.0;
            double af = is_last ? input.target_acceleration[d]
                                : (use_fast ? dp.accel_fast[s+1] : 0.0);

            trajs[s] = solve_and_sample_section(p0, v0, a0, pf, vf, af, dof_limits[d]);

            // Next section starts from the end state of this section
            if (!trajs[s].positions.empty()) {
                v0 = trajs[s].velocities.back();
                a0 = trajs[s].accelerations.back();
            } else {
                v0 = vf;
                a0 = af;
            }
        }
    }

    //! Get the state at a global time t from a sequence of section trajectories.
    //! Returns (p, v, a) and the cumulative path length s.
    void sample_section_seq(const std::vector<SectionTrajectory>& trajs,
                            double t,
                            double& p, double& v, double& a, double& s) const {
        double cum_t = 0.0;
        double cum_s = 0.0;
        for (size_t sec = 0; sec < trajs.size(); ++sec) {
            const auto& tr = trajs[sec];
            if (t <= cum_t + tr.duration + 1e-15) {
                double local_t = t - cum_t;
                local_t = std::min(std::max(local_t, 0.0), tr.duration);
                // Find the sample interval
                size_t idx = 0;
                for (size_t i = 0; i + 1 < tr.times.size(); ++i) {
                    if (tr.times[i+1] >= local_t - 1e-15) { idx = i; break; }
                    idx = i;
                }
                // Interpolate within this interval (or use exact if we have it)
                if (local_t >= tr.duration - 1e-15) {
                    p = tr.positions.back();
                    v = tr.velocities.back();
                    a = tr.accelerations.back();
                } else {
                    // Linear interpolation between samples
                    double t0 = tr.times[idx];
                    double t1 = tr.times[idx+1];
                    double frac = (t1 > t0 + 1e-15) ? (local_t - t0) / (t1 - t0) : 0.0;
                    frac = std::min(std::max(frac, 0.0), 1.0);
                    p = tr.positions[idx] + frac * (tr.positions[idx+1] - tr.positions[idx]);
                    v = tr.velocities[idx] + frac * (tr.velocities[idx+1] - tr.velocities[idx]);
                    a = tr.accelerations[idx] + frac * (tr.accelerations[idx+1] - tr.accelerations[idx]);
                }
                // Compute s: cumulative + local progress
                if (sec > 0) {
                    // Sum of previous section lengths
                    double p_start = trajs[sec].positions.front();
                    s = cum_s + std::abs(p - p_start);
                } else {
                    s = std::abs(p - trajs[0].positions.front());
                }
                return;
            }
            cum_t += tr.duration;
            cum_s += std::abs(tr.positions.back() - tr.positions.front());
        }
        // Past the end
        if (!trajs.empty()) {
            p = trajs.back().positions.back();
            v = trajs.back().velocities.back();
            a = trajs.back().accelerations.back();
            s = cum_s;
        } else {
            p = v = a = s = 0.0;
        }
    }

    //! Total duration of a sequence of section trajectories.
    double total_duration(const std::vector<SectionTrajectory>& trajs) const {
        double d = 0.0;
        for (const auto& t : trajs) d += t.duration;
        return d;
    }

    double state_to_u(size_t d, double p, double s) const {
        const auto& dp = dof_paths[d];
        if (dp.total_s < position_eps || total_u < position_eps) return 0.0;

        // Find which 1-D section s falls in
        size_t sec = 0;
        for (size_t k = 0; k + 1 < dp.cum_s.size(); ++k) {
            if (s <= dp.cum_s[k+1] + position_eps) { sec = k; break; }
            sec = k;
        }
        double seg_len = dp.seg_lengths[sec];
        double local_s = s - dp.cum_s[sec];
        double frac = (seg_len > position_eps) ? std::min(local_s / seg_len, 1.0) : 0.0;

        size_t wp_from = dp.extrema_wp[sec];
        size_t wp_to = dp.extrema_wp[sec + 1];

        // Map within the multi-dim waypoint segments
        double cum_seg = 0.0;
        double target = frac * seg_len;
        for (size_t j = wp_from; j < wp_to; ++j) {
            double seg_d = std::abs(wp_pos(d, j+1) - wp_pos(d, j));
            if (cum_seg + seg_d >= target - position_eps) {
                double f = (seg_d > position_eps) ? (target - cum_seg) / seg_d : 0.0;
                f = std::min(std::max(f, 0.0), 1.0);
                return cum_u[j] + f * (cum_u[j+1] - cum_u[j]);
            }
            cum_seg += seg_d;
        }
        return cum_u[wp_to];
    }

    double u_to_position(size_t d, double u) const {
        if (total_u < position_eps) return wp_pos(d, 0);
        size_t sec = 0;
        for (size_t k = 0; k + 1 < n_waypoints_; ++k) {
            if (u <= cum_u[k+1] + position_eps) { sec = k; break; }
            sec = k;
        }
        double su = cum_u[sec+1] - cum_u[sec];
        double f = (su > position_eps) ? std::min(std::max((u - cum_u[sec]) / su, 0.0), 1.0) : 0.0;
        return wp_pos(d, sec) + f * (wp_pos(d, sec+1) - wp_pos(d, sec));
    }

    //! Convert a tracked state on a monotonic 1-D section back to cumulative
    //! path length s. Unlike position_to_s(), this uses the active section
    //! chosen by the online tracker, so repeated positions across sections do
    //! not alias to the wrong branch during iterative u_ref updates.
    double tracked_state_to_s(size_t d, size_t section, double p) const {
        const auto& dp = dof_paths[d];
        if (dp.total_s < position_eps) return 0.0;
        if (section >= dp.seg_lengths.size()) return dp.total_s;

        const double p_start = dp.extrema_pos[section];
        const double p_end = dp.extrema_pos[section + 1];
        double local_s = (p_end >= p_start) ? (p - p_start) : (p_start - p);
        local_s = std::min(std::max(local_s, 0.0), dp.seg_lengths[section]);
        return std::min(std::max(dp.cum_s[section] + local_s, 0.0), dp.total_s);
    }

    //! Map a dt-sampled position back to cumulative s using the known forward
    //! path order from the current section. This avoids ambiguous global
    //! closest-section projection when a probe trajectory reaches the next
    //! section before dt elapses.
    double forward_position_to_s(size_t d, size_t section_hint, double p) const {
        const auto& dp = dof_paths[d];
        if (dp.total_s < position_eps) return 0.0;

        const size_t last_section = dp.seg_lengths.empty() ? 0 : (dp.seg_lengths.size() - 1);
        for (size_t section = std::min(section_hint, last_section); section < dp.seg_lengths.size(); ++section) {
            const double p_start = dp.extrema_pos[section];
            const double p_end = dp.extrema_pos[section + 1];
            const double lo = std::min(p_start, p_end) - overshoot_tolerance;
            const double hi = std::max(p_start, p_end) + overshoot_tolerance;
            if (p >= lo && p <= hi) {
                return tracked_state_to_s(d, section, p);
            }
        }

        return dp.total_s;
    }

    //! Map per-DoF cumulative path length s to position on the 1-D path.
    //! Within each section (between extrema) the path is monotonic, so
    //! position = p_start ± local_s depending on direction.
    double s_to_position(size_t d, double s) const {
        const auto& dp = dof_paths[d];
        if (dp.total_s < position_eps) return dp.extrema_pos[0];
        s = std::max(0.0, std::min(s, dp.total_s));

        size_t sec = 0;
        for (size_t k = 0; k + 1 < dp.cum_s.size(); ++k) {
            if (s <= dp.cum_s[k+1] + position_eps) { sec = k; break; }
            sec = k;
        }

        double local_s = std::max(0.0, std::min(s - dp.cum_s[sec], dp.seg_lengths[sec]));
        double p_start = dp.extrema_pos[sec];
        double p_end = dp.extrema_pos[sec + 1];
        double sign = (p_end >= p_start) ? 1.0 : -1.0;
        return p_start + sign * local_s;
    }

    //! Convert multi-dim path parameter u to per-DoF cumulative path length s.
    //! Traces through the multi-dim waypoint segments to find which 1-D section
    //! the progress corresponds to, then accumulates the per-DoF path length.
    double u_to_s_for_dof(size_t d, double u) const {
        const auto& dp = dof_paths[d];
        if (total_u < position_eps || dp.total_s < position_eps) return 0.0;
        u = std::max(0.0, std::min(u, total_u));

        // Find which multi-dim waypoint segment u falls in
        size_t wp_seg = 0;
        for (size_t k = 0; k + 1 < n_waypoints_; ++k) {
            if (u <= cum_u[k+1] + position_eps) { wp_seg = k; break; }
            wp_seg = k;
        }

        double su = cum_u[wp_seg + 1] - cum_u[wp_seg];
        double f = (su > position_eps) ? std::min(std::max((u - cum_u[wp_seg]) / su, 0.0), 1.0) : 0.0;

        // Find which 1-D section contains wp_seg
        size_t sec = 0;
        for (size_t k = 0; k + 1 < dp.extrema_wp.size(); ++k) {
            if (wp_seg < dp.extrema_wp[k + 1]) { sec = k; break; }
            sec = k;
        }

        // Accumulate per-DoF path length: full segments before wp_seg + fraction
        double s_within_section = 0.0;
        for (size_t j = dp.extrema_wp[sec]; j < wp_seg; ++j)
            s_within_section += std::abs(wp_pos(d, j + 1) - wp_pos(d, j));
        s_within_section += f * std::abs(wp_pos(d, wp_seg + 1) - wp_pos(d, wp_seg));

        return dp.cum_s[sec] + s_within_section;
    }

    double position_to_s(size_t d, double p) const {
        const auto& dp = dof_paths[d];
        if (dp.total_s < position_eps) return 0.0;

        double best_s = 0.0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t sec = 0; sec + 1 < dp.extrema_pos.size(); ++sec) {
            const double p_start = dp.extrema_pos[sec];
            const double p_end = dp.extrema_pos[sec + 1];
            const double lo = std::min(p_start, p_end);
            const double hi = std::max(p_start, p_end);
            const double p_clamped = std::min(std::max(p, lo), hi);
            const double dist = std::abs(p - p_clamped);
            const double s = dp.cum_s[sec] + std::abs(p_clamped - p_start);

            if (dist < best_dist) {
                best_dist = dist;
                best_s = s;
            }
        }

        return std::min(std::max(best_s, 0.0), dp.total_s);
    }

    struct TrackingResult {
        std::vector<std::vector<double>> positions;   // [dof][step]
        std::vector<std::vector<double>> velocities;
        std::vector<std::vector<double>> accelerations;
        std::vector<std::vector<double>> path_lengths;
        std::vector<std::vector<double>> u_values;
        //! Full Ruckig profile (intermediate solve) that governs the transition
        //! from step k−1 to step k, per DoF. Index [d][k] stores the profile
        //! consumed during sim step k (so [d][0] is unused). Reconstruction
        //! truncates each profile to sim_dt, preserving any phase transitions
        //! that fall inside the sim step — storing only j[0] would mis-represent
        //! sim steps where a phase boundary lies in [0, sim_dt).
        std::vector<std::vector<Profile>> profiles;
        size_t n_steps {0};
        double total_time {0.0};
        double max_deviation {0.0};
        double avg_deviation {0.0};
        double max_path_deviation {0.0};
        double avg_path_deviation {0.0};
    };

    double point_to_segment_distance(const std::vector<double>& point, size_t segment) const {
        double len_sq = 0.0;
        double dot_pa_ab = 0.0;
        std::vector<double> ab(degrees_of_freedom, 0.0);
        std::vector<double> pa(degrees_of_freedom, 0.0);

        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            const double a_d = wp_pos(d, segment);
            const double b_d = wp_pos(d, segment + 1);
            ab[d] = b_d - a_d;
            pa[d] = point[d] - a_d;
            len_sq += ab[d] * ab[d];
            dot_pa_ab += pa[d] * ab[d];
        }

        double t = 0.0;
        if (len_sq > position_eps) {
            t = std::min(std::max(dot_pa_ab / len_sq, 0.0), 1.0);
        }

        double sq = 0.0;
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            const double closest_d = wp_pos(d, segment) + t * ab[d];
            const double diff = point[d] - closest_d;
            sq += diff * diff;
        }

        return std::sqrt(sq);
    }

    //! Build a Ruckig-style Profile that represents exactly the first `dur`
    //! seconds of `src`, treating src's pre-trajectory brake and its main
    //! 7-phase body as a single logical sequence (brake phases first, then
    //! main phases). When Ruckig determines the input state violates a
    //! predictive limit (e.g. a > 0 and v_at_a_zero > v_max), it inserts a
    //! pre-trajectory brake sub-profile before the main profile — so
    //! Trajectory::at_time(t) samples the brake for t < brake.duration and
    //! the main profile for t ∈ [brake.duration, brake.duration + t_sum[6]].
    //! If we stored only the main profile (zeroing brake.duration), every
    //! sim-step that falls inside a brake window would emit a profile whose
    //! p[0] = post-brake state rather than the actual input state — creating
    //! position/velocity/acceleration jumps of up to the brake's traversal
    //! when the reconstructed trajectory is re-sampled.
    //!
    //! Strategy: concatenate brake (up to 2 phases) and main (up to 7 phases)
    //! into one linear list of (t,j) pairs, truncate to `dur`, and pack back
    //! into the 7-phase Profile layout. For typical Ruckig outputs the union
    //! of active phases is <= 7, so no information is lost; if the rare case
    //! of >7 active phases within a sim step arises, we truncate at dur
    //! before exhausting the list.
    static void truncate_profile(Profile& p, double dur) {
        // Gather (t, j) phases from brake, then main, skipping zero-duration.
        std::array<double, 9> pt{}, pj{};
        size_t n_phase = 0;
        double p0 {0.0}, v0 {0.0}, a0 {0.0};
        if (p.brake.duration > 0.0) {
            p0 = p.brake.p[0];
            v0 = p.brake.v[0];
            a0 = p.brake.a[0];
            for (size_t i = 0; i < 2; ++i) {
                if (p.brake.t[i] > 0.0) {
                    pt[n_phase] = p.brake.t[i];
                    pj[n_phase] = p.brake.j[i];
                    ++n_phase;
                }
            }
        } else {
            p0 = p.p[0];
            v0 = p.v[0];
            a0 = p.a[0];
        }
        for (size_t i = 0; i < 7; ++i) {
            if (p.t[i] > 0.0) {
                pt[n_phase] = p.t[i];
                pj[n_phase] = p.j[i];
                ++n_phase;
            }
        }

        // Truncate to `dur`.
        std::array<double, 7> kept_t{}, kept_j{};
        size_t n_kept = 0;
        double remaining = dur;
        for (size_t i = 0; i < n_phase && n_kept < 7; ++i) {
            if (pt[i] >= remaining - 1e-15) {
                kept_t[n_kept] = remaining;
                kept_j[n_kept] = pj[i];
                ++n_kept;
                remaining = 0.0;
                break;
            }
            kept_t[n_kept] = pt[i];
            kept_j[n_kept] = pj[i];
            ++n_kept;
            remaining -= pt[i];
        }
        if (remaining > 1e-15 && n_kept < 7) {
            // Merged profile shorter than dur → coast with zero jerk.
            kept_t[n_kept] = remaining;
            kept_j[n_kept] = 0.0;
            ++n_kept;
        }

        // Write back into the 7-phase Profile layout.
        for (size_t i = 0; i < 7; ++i) {
            p.t[i] = (i < n_kept) ? kept_t[i] : 0.0;
            p.j[i] = (i < n_kept) ? kept_j[i] : 0.0;
        }
        p.p[0] = p0;
        p.v[0] = v0;
        p.a[0] = a0;
        p.t_sum[0] = p.t[0];
        for (size_t i = 1; i < 7; ++i) p.t_sum[i] = p.t_sum[i-1] + p.t[i];
        for (size_t i = 0; i < 7; ++i) {
            p.a[i+1] = p.a[i] + p.t[i] * p.j[i];
            p.v[i+1] = p.v[i] + p.t[i] * (p.a[i] + p.t[i] * p.j[i] / 2.0);
            p.p[i+1] = p.p[i] + p.t[i] * (p.v[i] + p.t[i] * (p.a[i] / 2.0 + p.t[i] * p.j[i] / 6.0));
        }
        p.pf = p.p[7];
        p.vf = p.v[7];
        p.af = p.a[7];
        p.brake.duration = 0.0;
        p.brake.t[0] = 0.0; p.brake.t[1] = 0.0;
        p.accel.duration = 0.0;
        p.accel.t[0] = 0.0; p.accel.t[1] = 0.0;
    }

    //! Advance one DoF by exactly one simulation step (dt_goal) using the
    //! paper's online scheme (§III-C). Exactly ONE Ruckig solve is consumed
    //! per call so the returned first-phase jerk alone determines the full
    //! state transition — no mid-step section crossings, no mixing of jerks.
    //!
    //!   1. Solve upper trajectory to (p_end, 0, af_upper), sample at dt → s_U.
    //!   2. Solve lower trajectory to (p_end, 0, 0),        sample at dt → s_L.
    //!   3. Choose m ∈ [0,1] so that s_desired = s_L + m·(s_U - s_L) matches
    //!      the arc-length target s_target derived from u_ref.
    //!   4. Solve the intermediate trajectory to (p_end, 0, m·af_upper) and
    //!      sample at dt. This state is a Ruckig-trajectory prefix, so the
    //!      transition respects v_max, a_max, and j_max. The section index
    //!      is advanced if the sampled position reached p_end, but no further
    //!      advance is done in the same call.
    void advance_one_dof(size_t d,
                         double& p, double& v, double& a,
                         Profile& out_profile,
                         bool& out_profile_valid,
                         size_t& section,
                         bool& finished,
                         double dt_goal,
                         double s_target,
                         double target_v, double target_a,
                         double time_remaining_in_sim) {
        out_profile_valid = false;
        if (finished) return;

        const auto& dp = dof_paths[d];
        const auto& lim = dof_limits[d];

        // Skip any zero-length sections without consuming time.
        while (section + 1 < dp.extrema_pos.size()
               && std::abs(dp.extrema_pos[section + 1] - dp.extrema_pos[section])
                  < position_eps) {
            ++section;
        }
        if (section + 1 >= dp.extrema_pos.size()) {
            finished = true;
            return;
        }

        const double p_start = dp.extrema_pos[section];
        const double p_end   = dp.extrema_pos[section + 1];
        const bool is_last   = (section + 2 == dp.extrema_pos.size());
        const double sign = (p_end >= p_start) ? 1.0 : -1.0;

        if (is_last) {
            bool ok = solve_1dof(p, v, a, p_end, target_v, target_a, lim);
            if (!ok) ok = solve_1dof(p, v, a, p_end, 0.0, 0.0, lim);
            if (!ok) return;

            const double T = bs_traj.get_duration();
            out_profile = bs_traj.profiles[0][0];
            out_profile_valid = true;
            // Always sample at dt_goal so the tracked state matches what the
            // stored profile will produce under coast-after-end semantics.
            double pp, vv, aa;
            sample_traj(bs_traj, dt_goal, pp, vv, aa);
            p = pp; v = vv; a = aa;
            if (dt_goal >= T - 1e-12) {
                finished = true;
            }
            return;
        }

        const double af_upper_nominal = dp.accel_fast[section + 1];
        const double p_in = p, v_in = v, a_in = a;

        struct Candidate {
            bool ok {false};
            double s_dt {0.0};
            double p_dt {0.0};
            double v_dt {0.0};
            double a_dt {0.0};
            Profile profile;
            double duration {0.0};
        };

        auto sample_into = [&](Candidate& c) {
            c.ok = true;
            c.duration = bs_traj.get_duration();
            c.profile = bs_traj.profiles[0][0];
            sample_traj(bs_traj, dt_goal, c.p_dt, c.v_dt, c.a_dt);
            c.s_dt = forward_position_to_s(d, section, c.p_dt);
        };

        // Extend position bounds to include p_in so that a slight overshoot
        // from the previous section does not cause every trajectory to fail.
        const double bound_lo = std::min({p_in, p_start, p_end});
        const double bound_hi = std::max({p_in, p_start, p_end});

        // --- Upper (fast) trajectory: position control to (p_end, 0, af_fast).
        double af_upper = af_upper_nominal;
        Candidate upper{};
        if (solve_1dof(p_in, v_in, a_in, p_end, 0.0, af_upper, lim)
            && trajectory_stays_on_section(bs_traj, bound_lo, bound_hi, false)) {
            sample_into(upper);
        } else if (solve_1dof(p_in, v_in, a_in, p_end, 0.0, 0.0, lim)
                   && trajectory_stays_on_section(bs_traj, bound_lo, bound_hi, false)) {
            af_upper = 0.0;
            sample_into(upper);
        }

        // --- Lower (slow/brake) trajectory. Per paper §III-C: start with a
        // pure velocity-control brake (target v=0, a=0). Only if the brake
        // overshoots the section end, fall back to a position-control solve
        // with the smallest feasible positive target acceleration.
        Candidate lower{};
        double af_lower_used = 0.0;  // Track lower target acceleration for intermediate interpolation
        const double brake_bound_lo = bound_lo - overshoot_tolerance;
        const double brake_bound_hi = bound_hi + overshoot_tolerance;

        bool brake_valid = false;
        if (solve_brake(p_in, v_in, a_in, lim)) {
            const Profile& brake_prof = bs_traj.profiles[0][0];
            const Bound ext = brake_prof.get_position_extrema();
            if (ext.min >= brake_bound_lo && ext.max <= brake_bound_hi) {
                brake_valid = true;
                sample_into(lower);
                af_lower_used = 0.0;
            }
        }
        if (!brake_valid) {
            const double af_ref_mag = upper.ok ? std::abs(af_upper) : std::abs(af_upper_nominal);
            const double amin_mag = find_min_online_target_a(
                p_in, v_in, a_in, p_end, sign, lim, af_ref_mag
            );
            if (solve_1dof(p_in, v_in, a_in, p_end, 0.0, sign * amin_mag, lim)) {
                sample_into(lower);
                af_lower_used = sign * amin_mag;
            }
        }

        if (!upper.ok && !lower.ok) return;
        if (!upper.ok) { upper = lower; af_upper = af_lower_used; }
        if (!lower.ok) { lower = upper; af_lower_used = af_upper; }

        // --- Mapping factor m ∈ [0, 1] (paper Eq. 5).
        const double s_u = upper.s_dt;
        const double s_l = lower.s_dt;
        const double s_range = s_u - s_l;
        double m;
        if (s_range > position_eps) {
            m = (s_target - s_l) / s_range;
            m = std::max(0.0, std::min(1.0, m));
        } else {
            m = (s_target >= 0.5 * (s_u + s_l) - position_eps) ? 1.0 : 0.0;
        }

        // --- Intermediate trajectory (paper §III-C). For small dt, both
        // upper and lower are in their first jerk phase, and the kinematic
        // state is polynomial in the applied jerk. A convex combination of
        // upper's and lower's first-phase jerks produces the convex
        // combination of their Δt-sampled arc-lengths — precisely s_desired.
        auto first_phase_jerk = [](const Profile& pf) {
            if (pf.brake.duration > 0.0) {
                for (size_t i = 0; i < 2; ++i) {
                    if (pf.brake.t[i] > 0.0) return pf.brake.j[i];
                }
            }
            for (size_t i = 0; i < 7; ++i) {
                if (pf.t[i] > 0.0) return pf.j[i];
            }
            return 0.0;
        };
        const double j_u0 = first_phase_jerk(upper.profile);
        const double j_l0 = first_phase_jerk(lower.profile);
        const double j_mix = (1.0 - m) * j_l0 + m * j_u0;

        const double dt = dt_goal;
        p = p_in + v_in * dt + 0.5 * a_in * dt * dt
            + (j_mix * dt * dt * dt) / 6.0;
        v = v_in + a_in * dt + 0.5 * j_mix * dt * dt;
        a = a_in + j_mix * dt;

        // Build a synthetic single-phase profile for reconstruction.
        Profile mixed_profile;
        mixed_profile.t.fill(0.0);
        mixed_profile.j.fill(0.0);
        mixed_profile.t[0] = dt;
        mixed_profile.j[0] = j_mix;
        mixed_profile.p[0] = p_in;
        mixed_profile.v[0] = v_in;
        mixed_profile.a[0] = a_in;
        for (size_t i = 0; i < 7; ++i) {
            mixed_profile.a[i+1] = mixed_profile.a[i] + mixed_profile.t[i] * mixed_profile.j[i];
            mixed_profile.v[i+1] = mixed_profile.v[i]
                + mixed_profile.t[i] * (mixed_profile.a[i]
                                         + mixed_profile.t[i] * mixed_profile.j[i] / 2.0);
            mixed_profile.p[i+1] = mixed_profile.p[i]
                + mixed_profile.t[i] * (mixed_profile.v[i]
                                         + mixed_profile.t[i] * (mixed_profile.a[i] / 2.0
                                                                  + mixed_profile.t[i] * mixed_profile.j[i] / 6.0));
        }
        mixed_profile.t_sum[0] = mixed_profile.t[0];
        for (size_t i = 1; i < 7; ++i)
            mixed_profile.t_sum[i] = mixed_profile.t_sum[i-1] + mixed_profile.t[i];
        mixed_profile.pf = mixed_profile.p[7];
        mixed_profile.vf = mixed_profile.v[7];
        mixed_profile.af = mixed_profile.a[7];
        mixed_profile.brake.duration = 0.0;
        mixed_profile.brake.t[0] = 0.0; mixed_profile.brake.t[1] = 0.0;
        mixed_profile.accel.duration = 0.0;
        mixed_profile.accel.t[0] = 0.0; mixed_profile.accel.t[1] = 0.0;

        out_profile = mixed_profile;
        out_profile_valid = true;

        // Advance section when the upper trajectory completes within dt
        // (and we are tracking at full speed) or the position has reached
        // the section endpoint.
        const bool completed = upper.duration <= dt + 1e-12 && m >= 1.0 - 1e-9;
        const bool reached_pend = sign * (p - p_end) >= -overshoot_tolerance;
        if (completed || reached_pend) {
            // Snap state to prevent numerical overshoot from bleeding into
            // the next section.
            p = p_end;
            v = 0.0;
            ++section;
            if (section + 1 >= dp.extrema_pos.size()) finished = true;
        }
    }

    //! Run one tracking iteration using the online algorithm from §III-C/D.
    //!
    //! For every simulation step Δt and every DoF, we solve a fresh Ruckig
    //! upper/lower trajectory pair from the current state to the end of the
    //! current per-DoF section and mix their Δt-sampled states with the
    //! mapping factor m (Eq. 5) chosen so the per-DoF arc length matches
    //! s_target := u_to_s_for_dof(d, u_ref[k]).
    TrackingResult track(const InputParameter<DOFs, CustomVector>& input,
                         const std::vector<double>& u_ref,
                         const std::vector<std::vector<SectionTrajectory>>& /*fast_trajs*/) {
        const size_t n_steps = u_ref.size();
        TrackingResult res;
        res.n_steps = n_steps;
        res.total_time = (n_steps > 0 ? (n_steps - 1) : 0) * sim_dt;
        res.positions.resize(degrees_of_freedom);
        res.velocities.resize(degrees_of_freedom);
        res.accelerations.resize(degrees_of_freedom);
        res.path_lengths.resize(degrees_of_freedom);
        res.u_values.resize(degrees_of_freedom);
        res.profiles.resize(degrees_of_freedom);

        std::vector<double> p_cur(degrees_of_freedom), v_cur(degrees_of_freedom), a_cur(degrees_of_freedom);
        std::vector<size_t> section_cur(degrees_of_freedom, 0);
        std::vector<char> finished(degrees_of_freedom, 0);

        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            res.positions[d].resize(n_steps);
            res.velocities[d].resize(n_steps);
            res.accelerations[d].resize(n_steps);
            res.path_lengths[d].resize(n_steps);
            res.u_values[d].resize(n_steps);
            res.profiles[d].resize(n_steps);

            p_cur[d] = input.current_position[d];
            v_cur[d] = input.current_velocity[d];
            a_cur[d] = input.current_acceleration[d];
            res.positions[d][0]     = p_cur[d];
            res.velocities[d][0]    = v_cur[d];
            res.accelerations[d][0] = a_cur[d];
            res.path_lengths[d][0]  = 0.0;
            res.u_values[d][0]      = 0.0;
            finished[d] = (!input.enabled[d] || dof_paths[d].total_s < position_eps) ? 1 : 0;
        }

        for (size_t k = 1; k < n_steps; ++k) {
            const double time_remaining = (n_steps - k) * sim_dt;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (finished[d]) {
                    res.positions[d][k]     = p_cur[d];
                    res.velocities[d][k]    = v_cur[d];
                    res.accelerations[d][k] = a_cur[d];
                } else {
                    const double s_target = u_to_s_for_dof(d, u_ref[k]);
                    bool f = finished[d] != 0;
                    Profile step_prof;
                    bool prof_valid = false;
                    advance_one_dof(d, p_cur[d], v_cur[d], a_cur[d], step_prof, prof_valid,
                                    section_cur[d], f,
                                    sim_dt, s_target,
                                    input.target_velocity[d],
                                    input.target_acceleration[d],
                                    time_remaining);
                    finished[d] = f ? 1 : 0;
                    res.positions[d][k]     = p_cur[d];
                    res.velocities[d][k]    = v_cur[d];
                    res.accelerations[d][k] = a_cur[d];
                    if (prof_valid) res.profiles[d][k] = step_prof;
                }

                if (input.enabled[d] && dof_paths[d].total_s >= position_eps) {
                    const double s_cur = tracked_state_to_s(d, section_cur[d], p_cur[d]);
                    res.path_lengths[d][k] = s_cur;
                    res.u_values[d][k] = state_to_u(d, p_cur[d], s_cur);
                } else {
                    res.path_lengths[d][k] = 0.0;
                    res.u_values[d][k] = 0.0;
                }
            }
        }

        double sum_dev = 0.0, max_dev = 0.0;
        for (size_t k = 0; k < n_steps; ++k) {
            double sq = 0.0;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double diff = res.positions[d][k] - u_to_position(d, u_ref[k]);
                sq += diff * diff;
            }
            double dev = std::sqrt(sq);
            sum_dev += dev;
            max_dev = std::max(max_dev, dev);
        }
        res.max_deviation = max_dev;
        res.avg_deviation = n_steps > 0 ? sum_dev / n_steps : 0.0;

        double sum_path_dev = 0.0;
        double max_path_dev = 0.0;
        std::vector<double> point(degrees_of_freedom, 0.0);
        for (size_t k = 0; k < n_steps; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                point[d] = res.positions[d][k];
            }

            double best_dist = std::numeric_limits<double>::infinity();
            for (size_t seg = 0; seg + 1 < n_waypoints_; ++seg) {
                best_dist = std::min(best_dist, point_to_segment_distance(point, seg));
            }

            sum_path_dev += best_dist;
            max_path_dev = std::max(max_path_dev, best_dist);
        }
        res.max_path_deviation = max_path_dev;
        res.avg_path_deviation = n_steps > 0 ? sum_path_dev / n_steps : 0.0;
        return res;
    }

    void update_u_ref(std::vector<double>& u_ref, const TrackingResult& res) {
        if (res.n_steps < 2) return;

        const double delta = std::min(delta_u_ref, total_u);

        std::vector<double> pos_dev(res.n_steps, 0.0);
        for (size_t k = 0; k < res.n_steps; ++k) {
            double sq = 0.0;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                const double diff = res.positions[d][k] - u_to_position(d, u_ref[k]);
                sq += diff * diff;
            }
            pos_dev[k] = std::sqrt(sq);
        }

        // Paper §III-D updates the single path dimension and contiguous time
        // region outside the desired ±Δu_ref band that causes the largest
        // integrated position deviation. We check both undershoot (u < u_ref
        // - delta) and overshoot (u > u_ref + delta).
        size_t best_d = 0;
        size_t best_lo = 0, best_hi = 0;
        double best_area = 0.0;
        bool best_is_overshoot = false;

        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            // --- Undershoot regions (dimension is behind u_ref) ---
            bool in_seg = false;
            size_t seg_start = 0;
            for (size_t k = 0; k <= res.n_steps; ++k) {
                const double lower_bound = (k < res.n_steps) ? std::max(0.0, u_ref[k] - delta) : 0.0;
                const bool active = (k < res.n_steps && res.u_values[d][k] < lower_bound - position_eps);
                if (active && !in_seg) {
                    seg_start = k;
                    in_seg = true;
                } else if (!active && in_seg) {
                    double area = 0.0;
                    for (size_t j = seg_start; j < k; ++j) {
                        area += pos_dev[j];
                    }
                    if (area > best_area) {
                        best_area = area;
                        best_d = d;
                        best_lo = seg_start;
                        best_hi = k - 1;
                        best_is_overshoot = false;
                    }
                    in_seg = false;
                }
            }

            // --- Overshoot regions (dimension is ahead of u_ref) ---
            in_seg = false;
            for (size_t k = 0; k <= res.n_steps; ++k) {
                const double upper_bound = (k < res.n_steps) ? std::min(total_u, u_ref[k] + delta) : total_u;
                const bool active = (k < res.n_steps && res.u_values[d][k] > upper_bound + position_eps);
                if (active && !in_seg) {
                    seg_start = k;
                    in_seg = true;
                } else if (!active && in_seg) {
                    double area = 0.0;
                    for (size_t j = seg_start; j < k; ++j) {
                        area += pos_dev[j];
                    }
                    if (area > best_area) {
                        best_area = area;
                        best_d = d;
                        best_lo = seg_start;
                        best_hi = k - 1;
                        best_is_overshoot = true;
                    }
                    in_seg = false;
                }
            }
        }
        if (best_area <= 0.0) return;

        if (best_is_overshoot) {
            // Raise u_ref in the selected region so the overshooting
            // dimension lies within the upper u_ref band.
            for (size_t k = best_lo; k <= best_hi; ++k) {
                u_ref[k] = std::max(u_ref[k], std::max(res.u_values[best_d][k] - delta, 0.0));
            }
        } else {
            // Lower u_ref in the selected region so the undershooting
            // dimension lies within the lower u_ref band.
            for (size_t k = best_lo; k <= best_hi; ++k) {
                u_ref[k] = std::min(u_ref[k], std::min(res.u_values[best_d][k] + delta, total_u));
            }
        }

        // Keep u_ref monotonic after the local update.
        for (size_t k = 1; k < res.n_steps; ++k) {
            u_ref[k] = std::max(u_ref[k], u_ref[k - 1]);
            u_ref[k] = std::min(u_ref[k], total_u);
        }
        u_ref.back() = total_u;
    }

    template<bool throw_error>
    Result reconstruct(const TrackingResult& tracking,
                       const InputParameter<DOFs, CustomVector>& input,
                       Trajectory<DOFs, CustomVector>& traj) {
        if (tracking.n_steps < 2) {
            if constexpr (throw_error) {
                throw RuckigError("tracking produced fewer than 2 steps.");
            }
            return Result::Error;
        }

        const size_t n_micro = tracking.n_steps - 1;
        traj.degrees_of_freedom = degrees_of_freedom;
        traj.resize(n_micro - 1);
        traj.continue_calculation_counter = 0;

        // Each sim step is represented by the first sim_dt slice of the Ruckig
        // intermediate profile that generated the tracked transition. Copying
        // the profile and truncating to sim_dt preserves any phase transitions
        // that fall inside the sim step (e.g. jerk switching from ±j_max to 0
        // when acceleration saturates mid-step). Storing only the first-phase
        // jerk would mis-represent these steps and violate kinematic limits.
        const double dt = sim_dt;
        double cumulative = 0.0;
        for (size_t s = 0; s < n_micro; ++s) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                Profile& prof = traj.profiles[s][d];
                const Profile& src = tracking.profiles[d][s + 1];
                // If tracking produced a valid profile for this step, truncate
                // it to sim_dt. Otherwise (finished/disabled DoF), emit a rest.
                const bool has_profile =
                    src.t[0] > 0.0 || src.t[1] > 0.0 || src.t[2] > 0.0 ||
                    src.t[3] > 0.0 || src.t[4] > 0.0 || src.t[5] > 0.0 ||
                    src.t[6] > 0.0;
                if (has_profile) {
                    prof = src;
                    truncate_profile(prof, dt);
                } else {
                    prof.t.fill(0.0);
                    prof.j.fill(0.0);
                    prof.t[6] = dt;
                    prof.p[0] = tracking.positions[d][s];
                    prof.v[0] = tracking.velocities[d][s];
                    prof.a[0] = tracking.accelerations[d][s];
                    for (size_t i = 0; i < 7; ++i) {
                        prof.a[i+1] = prof.a[i] + prof.t[i] * prof.j[i];
                        prof.v[i+1] = prof.v[i] + prof.t[i] * (prof.a[i] + prof.t[i] * prof.j[i] / 2.0);
                        prof.p[i+1] = prof.p[i] + prof.t[i] * (prof.v[i] + prof.t[i] * (prof.a[i] / 2.0 + prof.t[i] * prof.j[i] / 6.0));
                    }
                    prof.t_sum[0] = prof.t[0];
                    for (size_t i = 1; i < 7; ++i) prof.t_sum[i] = prof.t_sum[i-1] + prof.t[i];
                    prof.pf = prof.p[7];
                    prof.vf = prof.v[7];
                    prof.af = prof.a[7];
                    prof.brake.duration = 0.0;
                    prof.brake.t[0] = 0.0; prof.brake.t[1] = 0.0;
                    prof.accel.duration = 0.0;
                    prof.accel.t[0] = 0.0; prof.accel.t[1] = 0.0;
                }
            }
            cumulative += dt;
            traj.cumulative_times[s] = cumulative;
        }

        traj.duration = cumulative;
        for (size_t d = 0; d < degrees_of_freedom; ++d)
            traj.independent_min_durations[d] = cumulative;

        return Result::Working;
    }

    bool needs_alignment_at_current(const InputParameter<DOFs, CustomVector>& input) const {
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!input.enabled[d]) continue;
            if (std::abs(input.current_velocity[d]) > position_eps
                || std::abs(input.current_acceleration[d]) > position_eps) {
                return true;
            }
        }

        return false;
    }

    bool needs_alignment_at_target(const InputParameter<DOFs, CustomVector>& input) const {
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!input.enabled[d]) continue;
            if (std::abs(input.target_velocity[d]) > position_eps
                || std::abs(input.target_acceleration[d]) > position_eps) {
                return true;
            }
        }

        return false;
    }

    InputParameter<DOFs, CustomVector> make_alignment_input(
            const InputParameter<DOFs, CustomVector>& input,
            const Vector<double>& current_position,
            const Vector<double>& current_velocity,
            const Vector<double>& current_acceleration,
            const Vector<double>& target_position,
            const Vector<double>& target_velocity,
            const Vector<double>& target_acceleration) const {
        InputParameter<DOFs, CustomVector> align_input = input;
        align_input.intermediate_positions.clear();
        align_input.per_section_max_velocity = std::nullopt;
        align_input.per_section_max_acceleration = std::nullopt;
        align_input.per_section_max_jerk = std::nullopt;
        align_input.per_section_min_velocity = std::nullopt;
        align_input.per_section_min_acceleration = std::nullopt;
        align_input.per_section_max_position = std::nullopt;
        align_input.per_section_min_position = std::nullopt;
        align_input.per_section_minimum_duration = std::nullopt;
        align_input.current_position = current_position;
        align_input.current_velocity = current_velocity;
        align_input.current_acceleration = current_acceleration;
        align_input.target_position = target_position;
        align_input.target_velocity = target_velocity;
        align_input.target_acceleration = target_acceleration;
        align_input.control_interface = ControlInterface::Position;
        align_input.synchronization = Synchronization::Time;
        align_input.duration_discretization = DurationDiscretization::Continuous;
        align_input.per_dof_control_interface = std::nullopt;
        align_input.per_dof_synchronization = std::nullopt;
        align_input.minimum_duration = std::nullopt;
        return align_input;
    }

    Vector<double> make_zero_vector() const {
        if constexpr (DOFs == 0) {
            Vector<double> v;
            v.resize(degrees_of_freedom);
            for (size_t d = 0; d < degrees_of_freedom; ++d) v[d] = 0.0;
            return v;
        } else {
            Vector<double> v {};
            for (size_t d = 0; d < degrees_of_freedom; ++d) v[d] = 0.0;
            return v;
        }
    }

    Trajectory<DOFs, CustomVector> make_temp_trajectory() const {
        if constexpr (DOFs == 0) {
            return Trajectory<DOFs, CustomVector>(degrees_of_freedom);
        } else {
            return Trajectory<DOFs, CustomVector>();
        }
    }

    size_t count_sections(const Trajectory<DOFs, CustomVector>& traj) const {
        return (traj.get_duration() > position_eps) ? traj.profiles.size() : 0;
    }

    size_t append_trajectory_sections(const Trajectory<DOFs, CustomVector>& src,
                                      Trajectory<DOFs, CustomVector>& dst,
                                      size_t dst_offset,
                                      double& time_offset) const {
        if (src.get_duration() <= position_eps) return dst_offset;

        for (size_t s = 0; s < src.profiles.size(); ++s) {
            dst.profiles[dst_offset] = src.profiles[s];
            dst.cumulative_times[dst_offset] = time_offset + src.cumulative_times[s];
            ++dst_offset;
        }

        time_offset += src.get_duration();
        return dst_offset;
    }

    void concatenate_trajectories(const Trajectory<DOFs, CustomVector>& pre_traj,
                                  const Trajectory<DOFs, CustomVector>& main_traj,
                                  const Trajectory<DOFs, CustomVector>& post_traj,
                                  Trajectory<DOFs, CustomVector>& traj) {
        const size_t n_pre = count_sections(pre_traj);
        const size_t n_main = count_sections(main_traj);
        const size_t n_post = count_sections(post_traj);
        const size_t total_sections = n_pre + n_main + n_post;

        traj.degrees_of_freedom = degrees_of_freedom;
        traj.resize((total_sections > 0) ? total_sections - 1 : 0);
        traj.continue_calculation_counter = 0;

        size_t dst_offset = 0;
        double time_offset = 0.0;
        dst_offset = append_trajectory_sections(pre_traj, traj, dst_offset, time_offset);
        dst_offset = append_trajectory_sections(main_traj, traj, dst_offset, time_offset);
        dst_offset = append_trajectory_sections(post_traj, traj, dst_offset, time_offset);

        traj.duration = time_offset;
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            traj.independent_min_durations[d] = time_offset;
        }
    }

    template<bool throw_error>
    Result calculate_core(const InputParameter<DOFs, CustomVector>& input,
                          Trajectory<DOFs, CustomVector>& traj,
                          double delta_time, bool& was_interrupted) {
        was_interrupted = false;
        n_waypoints_ = input.intermediate_positions.size() + 2;

        sim_dt = (delta_time > 0.0) ? std::min(0.0025, delta_time) : 0.0025;

        build_geometry(input);
        compute_accel_ranges(input);

        // Compute the per-DoF fast trajectories used to seed u_ref. The paper
        // assumes a small fixed discretization step Δt; in practice, some
        // sections can be much shorter than 2.5 ms. Adapting Δt to the
        // shortest non-trivial fast section avoids skipping over these short
        // sections during online tracking.
        std::vector<std::vector<SectionTrajectory>> fast_trajs(degrees_of_freedom);
        double slowest_t = 0.0;
        size_t slowest = 0;
        double min_fast_section_duration = std::numeric_limits<double>::infinity();

        auto compute_fast_trajectories = [&](double& out_slowest_t,
                                             size_t& out_slowest,
                                             double& out_min_fast_section_duration) {
            out_slowest_t = 0.0;
            out_slowest = 0;
            out_min_fast_section_duration = std::numeric_limits<double>::infinity();

            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (!input.enabled[d] || dof_paths[d].total_s < position_eps) {
                    SectionTrajectory trivial;
                    trivial.duration = 0.0;
                    trivial.times = {0.0};
                    trivial.positions = {dof_paths[d].extrema_pos[0]};
                    trivial.velocities = {0.0};
                    trivial.accelerations = {0.0};
                    fast_trajs[d] = {trivial};
                    continue;
                }

                compute_section_trajectories(d, input, true, fast_trajs[d]);

                const double dur = total_duration(fast_trajs[d]);
                if (dur > out_slowest_t) {
                    out_slowest_t = dur;
                    out_slowest = d;
                }

                for (const auto& sec: fast_trajs[d]) {
                    if (sec.duration > position_eps) {
                        out_min_fast_section_duration = std::min(out_min_fast_section_duration, sec.duration);
                    }
                }
            }
        };

        compute_fast_trajectories(slowest_t, slowest, min_fast_section_duration);

        if (std::isfinite(min_fast_section_duration)) {
            // Paper uses Δt = 2.5 ms. Cap the adaptive reduction so that the
            // lower/upper probes still produce a measurable s_u − s_l gap
            // within one step — otherwise the mapping factor m loses all
            // resolution and the tracker can't choose meaningfully between
            // brake and sprint.
            const double refined_sim_dt = std::min(sim_dt, std::max(1e-3, 0.25 * min_fast_section_duration));
            if (refined_sim_dt + 1e-15 < sim_dt) {
                sim_dt = refined_sim_dt;
                compute_fast_trajectories(slowest_t, slowest, min_fast_section_duration);
            }
        }
        std::fprintf(stderr, "[local] sim_dt=%.6f min_fast_sec_dur=%.6f slowest=%zu\n", sim_dt, min_fast_section_duration, slowest);

        if (slowest_t < sim_dt) {
            // Trivial: direct solve
            traj.degrees_of_freedom = degrees_of_freedom;
            traj.resize(0);
            traj.continue_calculation_counter = 0;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                segment_input.current_position[d] = input.current_position[d];
                segment_input.current_velocity[d] = input.current_velocity[d];
                segment_input.current_acceleration[d] = input.current_acceleration[d];
                segment_input.target_position[d] = input.target_position[d];
                segment_input.target_velocity[d] = input.target_velocity[d];
                segment_input.target_acceleration[d] = input.target_acceleration[d];
                segment_input.max_velocity[d] = input.max_velocity[d];
                segment_input.max_acceleration[d] = input.max_acceleration[d];
                segment_input.max_jerk[d] = input.max_jerk[d];
                segment_input.enabled[d] = input.enabled[d];
            }
            segment_input.min_velocity = input.min_velocity;
            segment_input.min_acceleration = input.min_acceleration;
            segment_input.control_interface = ControlInterface::Position;
            segment_input.synchronization = Synchronization::Time;
            segment_input.duration_discretization = DurationDiscretization::Continuous;
            segment_input.per_dof_control_interface = std::nullopt;
            segment_input.per_dof_synchronization = std::nullopt;
            segment_input.minimum_duration = std::nullopt;
            bool seg_interrupted = false;
            return segment_calc.template calculate<throw_error>(
                segment_input, traj, delta_time, seg_interrupted);
        }

        // Build u_ref from the slowest DoF's fast trajectory
        const size_t n_steps = static_cast<size_t>(std::ceil(slowest_t / sim_dt)) + 1;
        std::vector<double> u_ref(n_steps);
        for (size_t k = 0; k < n_steps; ++k) {
            double t = k * sim_dt;
            double p, v, a, s;
            sample_section_seq(fast_trajs[slowest], t, p, v, a, s);
            u_ref[k] = state_to_u(slowest, p, s);
        }
        for (size_t k = 1; k < n_steps; ++k)
            u_ref[k] = std::max(u_ref[k], u_ref[k-1]);
        u_ref.back() = total_u;

        // Paper §III-D iterative tracking. For 1-DoF the first tracking
        // result has zero deviation (u_ref == s), so no iteration is needed;
        // for multi-DoF we iterate and keep the iteration with the lowest
        // average deviation.
        TrackingResult latest = track(input, u_ref, fast_trajs);
        TrackingResult best = latest;
        std::vector<double> best_u_ref = u_ref;
        std::fprintf(stderr, "[local] sim_dt=%.6f n_steps=%zu total_u=%.4f\n", sim_dt, latest.n_steps, total_u);
        // Print d=0 at a few key indices
        auto dump = [&](const TrackingResult& r, const std::vector<double>& ur, int iter){
            size_t N = r.n_steps;
            size_t idx[6] = {0, N/5, 2*N/5, 3*N/5, 4*N/5, N-1};
            std::fprintf(stderr, "[local] iter=%d:\n", iter);
            for (size_t i : idx) {
                std::fprintf(stderr, "  k=%zu u_ref=%.4f | d0 p=%.4f u=%.4f | d1 p=%.4f u=%.4f | d2 p=%.4f u=%.4f\n",
                    i, ur[i],
                    r.positions[0][i], r.u_values[0][i],
                    r.positions[1][i], r.u_values[1][i],
                    r.positions[2][i], r.u_values[2][i]);
            }
        };
        dump(latest, u_ref, 0);
        if (degrees_of_freedom > 1) {
            for (int iter = 1; iter < n_tracking_iterations; ++iter) {
                update_u_ref(u_ref, latest);
                latest = track(input, u_ref, fast_trajs);
                if (iter <= 2 || iter == n_tracking_iterations - 1) dump(latest, u_ref, iter);
                if (latest.avg_path_deviation < best.avg_path_deviation) {
                    best = latest;
                    best_u_ref = u_ref;
                }
            }
            u_ref = best_u_ref;
        }

        return reconstruct<throw_error>(best, input, traj);
    }

public:
    size_t degrees_of_freedom;

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator():
        bs_calc(TargetCalculator<0, StandardVector>(1)),
        bs_input(InputParameter<0, StandardVector>(1)),
        bs_traj(Trajectory<0, StandardVector>(1)),
        bs_traj_upper(Trajectory<0, StandardVector>(1)),
        degrees_of_freedom(DOFs)
    { initialize_binary_search_input(); }

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t):
        LocalWaypointsCalculator() {}

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs):
        segment_calc(TargetCalculator<DOFs, CustomVector>(dofs)),
        segment_input(InputParameter<DOFs, CustomVector>(dofs)),
        segment_traj(Trajectory<DOFs, CustomVector>(dofs)),
        bs_calc(TargetCalculator<0, StandardVector>(1)),
        bs_input(InputParameter<0, StandardVector>(1)),
        bs_traj(Trajectory<0, StandardVector>(1)),
        bs_traj_upper(Trajectory<0, StandardVector>(1)),
        degrees_of_freedom(dofs)
    { initialize_binary_search_input(); }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs, size_t):
        LocalWaypointsCalculator(dofs) {}

    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input,
                     Trajectory<DOFs, CustomVector>& traj,
                     double delta_time, bool& was_interrupted) {
        const bool align_current = needs_alignment_at_current(input);
        const bool align_target = needs_alignment_at_target(input);

        if (!align_current && !align_target) {
            return calculate_core<throw_error>(input, traj, delta_time, was_interrupted);
        }

        InputParameter<DOFs, CustomVector> core_input = input;
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!input.enabled[d]) continue;
            core_input.current_velocity[d] = 0.0;
            core_input.current_acceleration[d] = 0.0;
            core_input.target_velocity[d] = 0.0;
            core_input.target_acceleration[d] = 0.0;
        }

        const Vector<double> zero_velocity = make_zero_vector();
        const Vector<double> zero_acceleration = make_zero_vector();

        Trajectory<DOFs, CustomVector> pre_traj = make_temp_trajectory();
        Trajectory<DOFs, CustomVector> main_traj = make_temp_trajectory();
        Trajectory<DOFs, CustomVector> post_traj = make_temp_trajectory();

        bool pre_interrupted = false;
        bool main_interrupted = false;
        bool post_interrupted = false;

        if (align_current) {
            // Pre-alignment: brake from (current_vel, current_acc) to v=0,
            // a=0 at (approximately) the current position. The braking
            // overshoot is small and the core trajectory then handles the
            // full path from start through all intermediate waypoints.
            Vector<double> pre_target = input.current_position;

            const auto pre_input = make_alignment_input(
                input,
                input.current_position,
                input.current_velocity,
                input.current_acceleration,
                pre_target,
                zero_velocity,
                zero_acceleration
            );

            const Result pre_res = segment_calc.template calculate<throw_error>(
                pre_input, pre_traj, delta_time, pre_interrupted
            );
            if (pre_res < 0) return pre_res;

            if (pre_traj.get_duration() > position_eps) {
                pre_traj.at_time(
                    pre_traj.get_duration(),
                    core_input.current_position,
                    core_input.current_velocity,
                    core_input.current_acceleration
                );
            }
        }

        const Result main_res = calculate_core<throw_error>(
            core_input, main_traj, delta_time, main_interrupted
        );
        if (main_res < 0) return main_res;

        if (align_target) {
            Vector<double> post_start_position = input.target_position;
            Vector<double> post_start_velocity = zero_velocity;
            Vector<double> post_start_acceleration = zero_acceleration;
            if (main_traj.get_duration() > position_eps) {
                main_traj.at_time(
                    main_traj.get_duration(),
                    post_start_position,
                    post_start_velocity,
                    post_start_acceleration
                );
            }

            const auto post_input = make_alignment_input(
                input,
                post_start_position,
                post_start_velocity,
                post_start_acceleration,
                input.target_position,
                input.target_velocity,
                input.target_acceleration
            );

            const Result post_res = segment_calc.template calculate<throw_error>(
                post_input, post_traj, delta_time, post_interrupted
            );
            if (post_res < 0) return post_res;
        }

        concatenate_trajectories(pre_traj, main_traj, post_traj, traj);
        was_interrupted = pre_interrupted || main_interrupted || post_interrupted;
        return Result::Working;
    }

    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>&,
                                 Trajectory<DOFs, CustomVector>&, double, bool&) {
        if constexpr (throw_error) {
            throw RuckigError("continue calculation not available in local waypoints backend.");
        }
        return Result::Error;
    }
};

} // namespace ruckig
