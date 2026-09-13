#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <iostream>
#include <vector>

#include <ruckig/block.hpp>
#include <ruckig/brake.hpp>
#include <ruckig/calculator_target.hpp>
#include <ruckig/error.hpp>
#include <ruckig/input_parameter.hpp>
#include <ruckig/position.hpp>
#include <ruckig/profile.hpp>
#include <ruckig/result.hpp>
#include <ruckig/trajectory.hpp>


namespace ruckig {

//! @brief Local (offline, network-free) calculator for trajectories along intermediate waypoints.
//!
//! The resulting trajectory passes exactly through every intermediate waypoint and consists of one
//! time-synchronized state-to-state section per pair of consecutive waypoints. The kinematic state
//! (velocity and acceleration of every DoF) at each intermediate waypoint is free and is chosen by an
//! iterative optimization of the total trajectory duration:
//!
//!   1. Structured initializations: stopping at every waypoint (which is always feasible and serves as
//!      the guaranteed upper bound), a path-tangent heuristic, and the time-optimal traversal of every
//!      DoF on its own.
//!   2. Local steps: a coordinate-wise pattern search on the (velocity, acceleration) pair of each DoF at
//!      each waypoint. Every candidate is evaluated by the exact synchronized duration of the two adjacent
//!      sections, using Ruckig's Step 1 (extremal profiles and blocked intervals).
//!   3. Global steps: random perturbations of the best solution followed by short local descents, in
//!      order to escape local minima of the (non-convex) problem.
//!   4. Selection and smoothing steps: among all local optima within a (by default zero) duration tolerance,
//!      the one with the lowest deviation from the piecewise-linear path through the waypoints is selected,
//!      and the deviation is further reduced by local adjustments within the remaining tolerance.
//!
//! Every section is finally calculated by the standard TargetCalculator, so all kinematic limits hold and
//! the position, velocity, and acceleration are continuous along the whole trajectory.
template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
class LocalWaypointsCalculator {
    template<class T> using Vector = CustomVector<T, DOFs>;

    using Clock = std::chrono::steady_clock;

    struct Limits {
        double v_max, v_min, a_max, a_min, j_max;
    };

    enum class Objective { Joint, Independent };

    constexpr static double duration_eps {1e-9};
    constexpr static double state_eps {1e-12};
    constexpr static size_t max_candidates {32};
    constexpr static size_t number_deviation_samples {24};

    // Problem data, filled by setup()
    size_t n_sections {0};
    std::vector<double> pos, vel, acc;  // [(n_sections + 1) * DoFs]: state at every waypoint (0: current, last: target)
    std::vector<Limits> limits;  // [n_sections * DoFs]
    std::vector<std::optional<double>> min_durations;  // [n_sections]
    std::vector<bool> enabled;  // [DoFs]
    std::vector<double> v_tangent;  // [(n_sections + 1) * DoFs]: heuristic pass-through velocity
    std::vector<double> unit_tangent;  // [(n_sections + 1) * DoFs]: unit tangent of the path at the waypoint
    std::vector<double> tangent_speed;  // [n_sections + 1]: maximum feasible speed along the unit tangent

    // Evaluation cache
    std::vector<Block> blocks;  // [n_sections * DoFs]
    std::vector<double> durations;  // [n_sections]
    std::vector<double> step_v, step_a;  // [(n_sections + 1) * DoFs]

    // Scratch
    std::vector<const Block*> row_prev, row_next;
    std::vector<double> sync_candidates;
    Block block_prev, block_next;
    std::vector<double> best_vel, best_acc;
    std::vector<double> independent_vel, independent_acc;

    //! Local optimum found during the optimization
    struct Solution {
        double duration;
        std::vector<double> vel, acc;
    };
    std::vector<Solution> solutions;
    constexpr static size_t max_solutions {64};

    // Optional deadline for soft interruption
    std::optional<Clock::time_point> deadline;
    bool deadline_reached {false};

    // Section calculation
    TargetCalculator<DOFs, CustomVector> section_calculator;
    InputParameter<DOFs, CustomVector> section_input;
    Trajectory<DOFs, CustomVector> section_trajectory;
    Vector<double> sample_position, sample_velocity, sample_acceleration;
    CustomVector<Bound, DOFs> position_extrema;


    inline size_t idx(size_t k, size_t d) const {
        return k * degrees_of_freedom + d;
    }

    bool check_deadline() {
        if (deadline && !deadline_reached && Clock::now() > *deadline) {
            deadline_reached = true;
        }
        return deadline_reached;
    }

    //! Calculate the extremal profiles of DoF d in section k for the given boundary states
    bool step1(size_t k, size_t d, double v0, double a0, double vf, double af, Block& block) const {
        const Limits& l = limits[idx(k, d)];

        Profile p;
        p.brake.get_position_brake_trajectory(v0, a0, l.v_max, l.v_min, l.a_max, l.a_min, l.j_max);
        p.set_boundary(pos[idx(k, d)], v0, a0, pos[idx(k + 1, d)], vf, af);
        p.brake.finalize(p.p[0], p.v[0], p.a[0]);

        PositionThirdOrderStep1 step {p.p[0], p.v[0], p.a[0], p.pf, p.vf, p.af, l.v_max, l.v_min, l.a_max, l.a_min, l.j_max};
        return step.get_profile(p, block);
    }

    bool step1(size_t k, size_t d, Block& block) const {
        return step1(k, d, vel[idx(k, d)], acc[idx(k, d)], vel[idx(k + 1, d)], acc[idx(k + 1, d)], block);
    }

    //! Time-synchronized duration of section k for the given extremal profiles of all DoFs (mirrors TargetCalculator::synchronize)
    std::optional<double> synchronize(size_t k, const Block* const* row) {
        double t_low = min_durations[k].value_or(0.0);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (enabled[d]) {
                t_low = std::max(t_low, row[d]->t_min);
            }
        }

        sync_candidates.clear();
        sync_candidates.push_back(t_low);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!enabled[d]) {
                continue;
            }
            if (row[d]->a && row[d]->a->right > t_low) {
                sync_candidates.push_back(row[d]->a->right);
            }
            if (row[d]->b && row[d]->b->right > t_low) {
                sync_candidates.push_back(row[d]->b->right);
            }
        }
        std::sort(sync_candidates.begin(), sync_candidates.end());

        for (const double t: sync_candidates) {
            if (std::isinf(t)) {
                continue;
            }

            bool is_blocked {false};
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (enabled[d] && row[d]->is_blocked(t)) {
                    is_blocked = true;
                    break;
                }
            }
            if (!is_blocked) {
                return t;
            }
        }
        return std::nullopt;
    }

    //! Recalculate all extremal profiles and section durations for the current waypoint states
    double recompute_all() {
        double total {0.0};
        for (size_t k = 0; k < n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (!enabled[d]) {
                    blocks[idx(k, d)].t_min = 0.0;
                    blocks[idx(k, d)].a = std::nullopt;
                    blocks[idx(k, d)].b = std::nullopt;
                    continue;
                }
                if (!step1(k, d, blocks[idx(k, d)])) {
                    return std::numeric_limits<double>::infinity();
                }
            }

            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                row_prev[d] = &blocks[idx(k, d)];
            }
            const auto t = synchronize(k, row_prev.data());
            if (!t) {
                return std::numeric_limits<double>::infinity();
            }
            durations[k] = *t;
            total += *t;
        }
        return total;
    }

    double total_duration() const {
        return std::accumulate(durations.begin(), durations.end(), 0.0);
    }

    //! Clip a waypoint state to the limits of both adjacent sections, including the velocity that will inevitably be reached
    void clip_state(size_t k, size_t d, double& v, double& a) const {
        const Limits& lp = limits[idx(k - 1, d)];
        const Limits& ln = limits[idx(k, d)];

        v = std::clamp(v, std::max(lp.v_min, ln.v_min), std::min(lp.v_max, ln.v_max));
        a = std::clamp(a, std::max(lp.a_min, ln.a_min), std::min(lp.a_max, ln.a_max));

        // Start of the next section: the velocity will inevitably rise to v + a^2/(2j) before the acceleration can reach zero.
        // Target of the previous section: the velocity has inevitably been v - a^2/(2j) before. Both need to stay within the limits.
        if (a > 0.0) {
            a = std::min(a, std::sqrt(std::max(0.0, 2 * ln.j_max * (ln.v_max - v))));
            a = std::min(a, std::sqrt(std::max(0.0, 2 * lp.j_max * (v - lp.v_min))));
            for (size_t i = 0; i < 64 && (v + a * a / (2 * ln.j_max) > ln.v_max || v - a * a / (2 * lp.j_max) < lp.v_min); ++i) {
                a *= 0.999;
            }
        } else if (a < 0.0) {
            a = std::max(a, -std::sqrt(std::max(0.0, 2 * ln.j_max * (v - ln.v_min))));
            a = std::max(a, -std::sqrt(std::max(0.0, 2 * lp.j_max * (lp.v_max - v))));
            for (size_t i = 0; i < 64 && (v - a * a / (2 * ln.j_max) < ln.v_min || v + a * a / (2 * lp.j_max) > lp.v_max); ++i) {
                a *= 0.999;
            }
        }

        if (std::abs(v) < state_eps) v = 0.0;
        if (std::abs(a) < state_eps) a = 0.0;
    }

    double v_upper(size_t k, size_t d) const { return std::min(limits[idx(k - 1, d)].v_max, limits[idx(k, d)].v_max); }
    double v_lower(size_t k, size_t d) const { return std::max(limits[idx(k - 1, d)].v_min, limits[idx(k, d)].v_min); }
    double a_upper(size_t k, size_t d) const { return std::min(limits[idx(k - 1, d)].a_max, limits[idx(k, d)].a_max); }
    double a_lower(size_t k, size_t d) const { return std::max(limits[idx(k - 1, d)].a_min, limits[idx(k, d)].a_min); }

    //! Second-largest minimum duration among the DoFs of a section (for breaking ties between co-limiting DoFs)
    double second_largest(const Block* const* row) const {
        double first {0.0}, second {0.0};
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!enabled[d]) continue;
            const double t = row[d]->t_min;
            if (t > first) {
                second = first;
                first = t;
            } else if (t > second) {
                second = t;
            }
        }
        return second;
    }

    //! Evaluate a candidate state of DoF d at waypoint k. Returns the extremal profiles and durations of both adjacent sections.
    bool evaluate(size_t k, size_t d, double v, double a, Block& b_prev, Block& b_next, double& t_prev, double& t_next, double* tie = nullptr) {
        if (!step1(k - 1, d, vel[idx(k - 1, d)], acc[idx(k - 1, d)], v, a, b_prev)) {
            return false;
        }
        if (!step1(k, d, v, a, vel[idx(k + 1, d)], acc[idx(k + 1, d)], b_next)) {
            return false;
        }

        for (size_t dd = 0; dd < degrees_of_freedom; ++dd) {
            row_prev[dd] = &blocks[idx(k - 1, dd)];
            row_next[dd] = &blocks[idx(k, dd)];
        }
        row_prev[d] = &b_prev;
        row_next[d] = &b_next;

        const auto tp = synchronize(k - 1, row_prev.data());
        if (!tp) {
            return false;
        }
        const auto tn = synchronize(k, row_next.data());
        if (!tn) {
            return false;
        }

        t_prev = *tp;
        t_next = *tn;
        if (tie) {
            *tie = second_largest(row_prev.data()) + second_largest(row_next.data());
        }
        return true;
    }

    //! Objective for the state of a single DoF at a single waypoint: the duration of both adjacent sections.
    //! Ties between co-limiting DoFs are broken by the second-largest DoF durations of the sections.
    struct Score {
        double primary, secondary;

        bool better_than(const Score& other) const {
            if (primary < other.primary - duration_eps) return true;
            if (primary > other.primary + duration_eps) return false;
            return secondary < other.secondary - duration_eps;
        }
    };

    static Score score(double t_prev, double t_next, double tie, const Block& b_prev, const Block& b_next, Objective objective) {
        if (objective == Objective::Independent) {
            return {b_prev.t_min + b_next.t_min, 0.0};
        }
        return {t_prev + t_next, tie};
    }

    //! Pattern search on the state of DoF d at waypoint k. Returns true if the state was improved.
    bool optimize_waypoint_dof(size_t k, size_t d, Objective objective) {
        const size_t i = idx(k, d);
        const double v_cur = vel[i], a_cur = acc[i];
        const double dv = step_v[i], da = step_a[i];

        std::array<std::pair<double, double>, max_candidates> candidates;
        size_t n_candidates {0};
        auto add = [&](double v, double a) {
            clip_state(k, d, v, a);
            for (size_t c = 0; c < n_candidates; ++c) {
                if (std::abs(candidates[c].first - v) < state_eps && std::abs(candidates[c].second - a) < state_eps) {
                    return;
                }
            }
            if (n_candidates < max_candidates) {
                candidates[n_candidates++] = {v, a};
            }
        };

        add(v_cur, a_cur);
        add(v_cur + dv, a_cur);
        add(v_cur - dv, a_cur);
        add(v_cur, a_cur + da);
        add(v_cur, a_cur - da);
        add(v_cur + dv, a_cur + da);
        add(v_cur + dv, a_cur - da);
        add(v_cur - dv, a_cur + da);
        add(v_cur - dv, a_cur - da);
        add(0.0, 0.0);
        add(v_cur, 0.0);
        add(0.0, a_cur);
        add(v_upper(k, d), 0.0);
        add(v_lower(k, d), 0.0);
        add(v_cur, a_upper(k, d));
        add(v_cur, a_lower(k, d));
        add(v_tangent[i], 0.0);

        // Current state
        double t_prev, t_next, tie;
        if (!evaluate(k, d, v_cur, a_cur, block_prev, block_next, t_prev, t_next, &tie)) {
            return false;
        }
        Score best = score(t_prev, t_next, tie, block_prev, block_next, objective);
        std::optional<size_t> best_candidate;

        for (size_t c = 1; c < n_candidates; ++c) {
            const auto [v, a] = candidates[c];
            if (!evaluate(k, d, v, a, block_prev, block_next, t_prev, t_next, &tie)) {
                continue;
            }
            const Score s = score(t_prev, t_next, tie, block_prev, block_next, objective);
            if (s.better_than(best)) {
                best = s;
                best_candidate = c;
            }
        }

        if (!best_candidate) {
            return false;
        }

        // Commit the best candidate
        const auto [v, a] = candidates[*best_candidate];
        evaluate(k, d, v, a, block_prev, block_next, t_prev, t_next);
        vel[i] = v;
        acc[i] = a;
        blocks[idx(k - 1, d)] = block_prev;
        blocks[idx(k, d)] = block_next;
        durations[k - 1] = t_prev;
        durations[k] = t_next;
        return true;
    }

    void reset_steps(double fraction) {
        for (size_t k = 1; k + 1 <= n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                step_v[idx(k, d)] = fraction * std::max(v_upper(k, d) - v_lower(k, d), state_eps);
                step_a[idx(k, d)] = fraction * std::max(a_upper(k, d) - a_lower(k, d), state_eps);
            }
        }
    }

    //! Coordinate descent over the states of the waypoints [k_first, k_last] with shrinking step sizes
    void local_descent(size_t max_sweeps, double initial_step_fraction, Objective objective, size_t k_first = 1, size_t k_last = std::numeric_limits<size_t>::max()) {
        reset_steps(initial_step_fraction);
        k_first = std::max<size_t>(k_first, 1);
        k_last = std::min<size_t>(k_last, n_sections - 1);
        if (k_first > k_last) {
            return;
        }

        double step_fraction = initial_step_fraction;
        for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            bool improved {false};
            const bool forward = (sweep % 2 == 0);
            for (size_t kk = k_first; kk <= k_last; ++kk) {
                const size_t k = forward ? kk : k_last + k_first - kk;
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (enabled[d]) {
                        improved |= optimize_waypoint_dof(k, d, objective);
                    }
                }
            }

            if (debug && objective == Objective::Joint) {
                std::cout << "[ruckig] waypoints:   sweep " << sweep << ": " << total_duration() << (improved ? "" : " (no improvement)") << std::endl;
            }

            if (check_deadline()) {
                return;
            }

            if (!improved) {
                step_fraction *= 0.5;
                if (step_fraction < 1e-4) {
                    return;
                }
                for (auto& s: step_v) s *= 0.5;
                for (auto& s: step_a) s *= 0.5;
            }
        }
    }

    void save_best() {
        best_vel = vel;
        best_acc = acc;
    }

    //! Remember the current states as a candidate solution (if not known yet)
    void remember_solution() {
        const double duration = total_duration();
        for (const auto& s: solutions) {
            if (std::abs(s.duration - duration) < duration_eps) {
                return;
            }
        }
        if (solutions.size() < max_solutions) {
            solutions.push_back({duration, vel, acc});
        }
    }

    //! Maximum and mean deviation of the current states from the piecewise-linear path (upper bound, uses only the own section)
    bool solution_deviation(const InputParameter<DOFs, CustomVector>& input, double delta_time, double& max_deviation, double& mean_deviation) {
        max_deviation = 0.0;
        mean_deviation = 0.0;
        for (size_t k = 0; k < n_sections; ++k) {
            if (!compute_section(input, k, delta_time)) {
                return false;
            }
            double section_max, section_mean;
            section_deviation(k, section_max, section_mean);
            max_deviation = std::max(max_deviation, section_max);
            mean_deviation += section_mean * section_trajectory.get_duration();
        }
        const double total = total_duration();
        mean_deviation = (total > 0.0) ? mean_deviation / total : 0.0;
        return true;
    }

    void restore_best() {
        vel = best_vel;
        acc = best_acc;
    }

    void set_zero_states() {
        for (size_t k = 1; k < n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                vel[idx(k, d)] = 0.0;
                acc[idx(k, d)] = 0.0;
            }
        }
    }

    void set_tangent_states(double scale) {
        for (size_t k = 1; k < n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double v = scale * v_tangent[idx(k, d)];
                double a = 0.0;
                clip_state(k, d, v, a);
                vel[idx(k, d)] = v;
                acc[idx(k, d)] = a;
            }
        }
    }

    //! Heuristic velocity for passing through the waypoints along the path
    void compute_tangent_velocities() {
        for (size_t k = 1; k < n_sections; ++k) {
            // Unit tangent: mean of the normalized directions of both adjacent segments
            double len_in {0.0}, len_out {0.0};
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (!enabled[d]) continue;
                len_in += std::pow(pos[idx(k, d)] - pos[idx(k - 1, d)], 2);
                len_out += std::pow(pos[idx(k + 1, d)] - pos[idx(k, d)], 2);
            }
            len_in = std::sqrt(len_in);
            len_out = std::sqrt(len_out);
            double norm {0.0};
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double t {0.0};
                if (enabled[d]) {
                    if (len_in > state_eps) t += (pos[idx(k, d)] - pos[idx(k - 1, d)]) / len_in;
                    if (len_out > state_eps) t += (pos[idx(k + 1, d)] - pos[idx(k, d)]) / len_out;
                }
                unit_tangent[idx(k, d)] = t;
                norm += t * t;
            }
            norm = std::sqrt(norm);
            double speed = std::numeric_limits<double>::infinity();
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                unit_tangent[idx(k, d)] = (norm > state_eps) ? unit_tangent[idx(k, d)] / norm : 0.0;
                const double t = unit_tangent[idx(k, d)];
                if (std::abs(t) > state_eps) {
                    speed = std::min(speed, ((t > 0.0) ? v_upper(k, d) : -v_lower(k, d)) / std::abs(t));
                }
            }
            tangent_speed[k] = std::isinf(speed) ? 0.0 : speed;
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                const double dp_in = pos[idx(k, d)] - pos[idx(k - 1, d)];
                const double dp_out = pos[idx(k + 1, d)] - pos[idx(k, d)];
                double v {0.0};
                if (dp_in * dp_out > 0.0) {
                    const Limits& lp = limits[idx(k - 1, d)];
                    const Limits& ln = limits[idx(k, d)];
                    const double a_lim = std::min({lp.a_max, ln.a_max, -lp.a_min, -ln.a_min});
                    const double v_lim = (dp_out > 0.0) ? std::min(lp.v_max, ln.v_max) : std::min(-lp.v_min, -ln.v_min);
                    const double v_reach = std::sqrt(a_lim * std::min(std::abs(dp_in), std::abs(dp_out)));
                    v = ((dp_out > 0.0) ? 1.0 : -1.0) * std::min(v_lim, v_reach);
                }
                double a {0.0};
                clip_state(k, d, v, a);
                v_tangent[idx(k, d)] = v;
            }
        }
    }

    void setup(const InputParameter<DOFs, CustomVector>& input) {
        const size_t n_waypoints = input.intermediate_positions.size();
        n_sections = n_waypoints + 1;
        const size_t n_states = (n_sections + 1) * degrees_of_freedom;

        pos.resize(n_states);
        vel.resize(n_states);
        acc.resize(n_states);
        v_tangent.resize(n_states);
        unit_tangent.resize(n_states);
        tangent_speed.resize(n_sections + 1);
        step_v.resize(n_states);
        step_a.resize(n_states);
        limits.resize(n_sections * degrees_of_freedom);
        blocks.resize(n_sections * degrees_of_freedom);
        min_durations.resize(n_sections);
        durations.resize(n_sections);
        enabled.resize(degrees_of_freedom);
        row_prev.resize(degrees_of_freedom);
        row_next.resize(degrees_of_freedom);
        sync_candidates.reserve(2 * degrees_of_freedom + 1);

        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            enabled[d] = input.enabled[d];

            pos[idx(0, d)] = input.current_position[d];
            vel[idx(0, d)] = input.current_velocity[d];
            acc[idx(0, d)] = input.current_acceleration[d];
            for (size_t k = 1; k < n_sections; ++k) {
                pos[idx(k, d)] = input.intermediate_positions[k - 1][d];
                vel[idx(k, d)] = 0.0;
                acc[idx(k, d)] = 0.0;
            }
            pos[idx(n_sections, d)] = input.target_position[d];
            vel[idx(n_sections, d)] = input.target_velocity[d];
            acc[idx(n_sections, d)] = input.target_acceleration[d];

            for (size_t k = 0; k < n_sections; ++k) {
                Limits& l = limits[idx(k, d)];
                l.v_max = input.per_section_max_velocity ? input.per_section_max_velocity.value()[k][d] : input.max_velocity[d];
                l.a_max = input.per_section_max_acceleration ? input.per_section_max_acceleration.value()[k][d] : input.max_acceleration[d];
                l.j_max = input.per_section_max_jerk ? input.per_section_max_jerk.value()[k][d] : input.max_jerk[d];
                if (input.per_section_min_velocity) {
                    l.v_min = input.per_section_min_velocity.value()[k][d];
                } else {
                    l.v_min = input.min_velocity ? input.min_velocity.value()[d] : -l.v_max;
                }
                if (input.per_section_min_acceleration) {
                    l.a_min = input.per_section_min_acceleration.value()[k][d];
                } else {
                    l.a_min = input.min_acceleration ? input.min_acceleration.value()[d] : -l.a_max;
                }
            }
        }

        for (size_t k = 0; k < n_sections; ++k) {
            min_durations[k] = std::nullopt;
            if (input.per_section_minimum_duration && k < input.per_section_minimum_duration->size()) {
                const double t = input.per_section_minimum_duration.value()[k];
                if (t > 0.0) {
                    min_durations[k] = t;
                }
            }
        }

        compute_tangent_velocities();
    }

    //! Fill the input of a single section from the current waypoint states
    void fill_section_input(const InputParameter<DOFs, CustomVector>& input, size_t k) {
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            section_input.current_position[d] = pos[idx(k, d)];
            section_input.current_velocity[d] = vel[idx(k, d)];
            section_input.current_acceleration[d] = acc[idx(k, d)];
            section_input.target_position[d] = pos[idx(k + 1, d)];
            section_input.target_velocity[d] = vel[idx(k + 1, d)];
            section_input.target_acceleration[d] = acc[idx(k + 1, d)];

            const Limits& l = limits[idx(k, d)];
            section_input.max_velocity[d] = l.v_max;
            section_input.max_acceleration[d] = l.a_max;
            section_input.max_jerk[d] = l.j_max;
            section_input.min_velocity.value()[d] = l.v_min;
            section_input.min_acceleration.value()[d] = l.a_min;
            section_input.enabled[d] = input.enabled[d];
        }
        section_input.control_interface = ControlInterface::Position;
        section_input.synchronization = input.synchronization;
        section_input.duration_discretization = DurationDiscretization::Continuous;
        section_input.minimum_duration = min_durations[k];
    }

    bool compute_section(const InputParameter<DOFs, CustomVector>& input, size_t k, double delta_time) {
        fill_section_input(input, k);
        bool interrupted {false};
        return section_calculator.template calculate<false>(section_input, section_trajectory, delta_time, interrupted) == Result::Working;
    }

    //! Maximum and mean deviation of the (just calculated) section k from the straight line between its waypoints
    void section_deviation(size_t k, double& max_deviation, double& mean_deviation) {
        max_deviation = 0.0;
        mean_deviation = 0.0;
        const double duration = section_trajectory.get_duration();
        if (duration <= 0.0) {
            return;
        }

        double length_sq {0.0};
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (enabled[d]) {
                const double diff = pos[idx(k + 1, d)] - pos[idx(k, d)];
                length_sq += diff * diff;
            }
        }

        double sum {0.0}, max_sq {0.0};
        for (size_t s = 0; s < number_deviation_samples; ++s) {
            const double t = duration * (s + 0.5) / number_deviation_samples;
            section_trajectory.at_time(t, sample_position);

            double projection {0.0};
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (enabled[d]) {
                    projection += (sample_position[d] - pos[idx(k, d)]) * (pos[idx(k + 1, d)] - pos[idx(k, d)]);
                }
            }
            const double h = (length_sq > state_eps) ? std::clamp(projection / length_sq, 0.0, 1.0) : 0.0;

            double dist_sq {0.0};
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (enabled[d]) {
                    const double diff = sample_position[d] - pos[idx(k, d)] - h * (pos[idx(k + 1, d)] - pos[idx(k, d)]);
                    dist_sq += diff * diff;
                }
            }
            sum += std::sqrt(dist_sq);
            max_sq = std::max(max_sq, dist_sq);
        }
        max_deviation = std::sqrt(max_sq);
        mean_deviation = sum / number_deviation_samples;
    }

    //! Deviation cost of the (just calculated) section k used in the smoothing steps
    double section_deviation_cost(size_t k) {
        double max_deviation, mean_deviation;
        section_deviation(k, max_deviation, mean_deviation);
        return max_deviation * max_deviation + mean_deviation * mean_deviation;
    }

    //! Accept a smoothing candidate: any improvement if the duration does not increase, otherwise only a substantial one
    static bool accept_smoothing_candidate(double cost, double duration, double best_cost, double duration_current) {
        constexpr double required_improvement {0.1};
        if (duration > duration_current + duration_eps) {
            return cost < best_cost * (1.0 - required_improvement) - 1e-14;
        }
        return cost < best_cost - 1e-14;
    }

    //! Evaluate deviation and duration of the two sections adjacent to waypoint k
    bool evaluate_smoothing(const InputParameter<DOFs, CustomVector>& input, size_t k, double delta_time, double& cost, double& duration) {
        cost = 0.0;
        duration = 0.0;
        for (const size_t s: {k - 1, k}) {
            if (!compute_section(input, s, delta_time)) {
                return false;
            }
            duration += section_trajectory.get_duration();
            cost += section_deviation_cost(s);
        }
        return true;
    }

    //! Reduce the path deviation of the state of DoF d at waypoint k. The duration of both adjacent sections may increase by at most the given budget.
    bool smooth_waypoint_dof(const InputParameter<DOFs, CustomVector>& input, size_t k, size_t d, double delta_time, double step_fraction, double& duration_budget) {
        const size_t i = idx(k, d);
        const double v_cur = vel[i], a_cur = acc[i];
        const double dv = step_fraction * std::max(v_upper(k, d) - v_lower(k, d), state_eps);
        const double da = step_fraction * std::max(a_upper(k, d) - a_lower(k, d), state_eps);

        double best_cost, best_duration;
        if (!evaluate_smoothing(input, k, delta_time, best_cost, best_duration)) {
            return false;
        }
        const double duration_current = best_duration;
        const double duration_limit = best_duration + duration_budget + duration_eps;

        std::array<std::pair<double, double>, max_candidates> candidates;
        size_t n_candidates {0};
        auto add = [&](double v, double a) {
            clip_state(k, d, v, a);
            for (size_t c = 0; c < n_candidates; ++c) {
                if (std::abs(candidates[c].first - v) < state_eps && std::abs(candidates[c].second - a) < state_eps) {
                    return;
                }
            }
            if (n_candidates < max_candidates) {
                candidates[n_candidates++] = {v, a};
            }
        };
        add(v_cur, a_cur);
        add(v_cur + dv, a_cur);
        add(v_cur - dv, a_cur);
        add(v_cur, a_cur + da);
        add(v_cur, a_cur - da);
        add(v_cur + dv, a_cur + da);
        add(v_cur + dv, a_cur - da);
        add(v_cur - dv, a_cur + da);
        add(v_cur - dv, a_cur - da);
        add(v_tangent[i], 0.0);
        add(v_tangent[i], a_cur);
        add(0.5 * (v_cur + v_tangent[i]), 0.5 * a_cur);
        add(v_cur, 0.5 * a_cur);
        add(v_cur, 0.0);
        add(0.0, 0.0);
        for (const double fraction: {0.25, 0.5, 0.75, 1.0}) {
            add(fraction * tangent_speed[k] * unit_tangent[i], 0.0);
        }

        std::optional<size_t> best_candidate;
        for (size_t c = 1; c < n_candidates; ++c) {
            vel[i] = candidates[c].first;
            acc[i] = candidates[c].second;

            double cost, duration;
            if (!evaluate_smoothing(input, k, delta_time, cost, duration) || duration > duration_limit) {
                continue;
            }
            if (accept_smoothing_candidate(cost, duration, best_cost, duration_current)) {
                best_cost = cost;
                best_duration = duration;
                best_candidate = c;
            }
        }

        if (!best_candidate) {
            vel[i] = v_cur;
            acc[i] = a_cur;
            return false;
        }

        vel[i] = candidates[*best_candidate].first;
        acc[i] = candidates[*best_candidate].second;
        duration_budget = std::max(0.0, duration_budget - (best_duration - duration_current));

        // Keep the evaluation cache consistent
        double t_prev, t_next;
        if (evaluate(k, d, vel[i], acc[i], block_prev, block_next, t_prev, t_next)) {
            blocks[idx(k - 1, d)] = block_prev;
            blocks[idx(k, d)] = block_next;
            durations[k - 1] = t_prev;
            durations[k] = t_next;
        }
        return true;
    }

    //! Reduce the path deviation by scaling the states of all DoFs at waypoint k jointly (e.g. slowing down at a corner)
    bool smooth_waypoint_joint(const InputParameter<DOFs, CustomVector>& input, size_t k, double delta_time, double& duration_budget) {
        std::vector<double> v_cur(degrees_of_freedom), a_cur(degrees_of_freedom);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            v_cur[d] = vel[idx(k, d)];
            a_cur[d] = acc[idx(k, d)];
        }

        double best_cost, best_duration;
        if (!evaluate_smoothing(input, k, delta_time, best_cost, best_duration)) {
            return false;
        }
        const double duration_current = best_duration;
        const double duration_limit = best_duration + duration_budget + duration_eps;

        // Candidates: scaled current states, and velocities proportional to the path tangent (straight passage)
        constexpr std::array<std::pair<double, double>, 8> scales {{{0.75, 0.75}, {0.5, 0.5}, {0.25, 0.25}, {0.0, 0.0}, {0.75, 1.0}, {0.5, 1.0}, {1.0, 0.5}, {1.0, 0.0}}};
        constexpr std::array<double, 4> fractions {{0.25, 0.5, 0.75, 1.0}};
        auto set_candidate = [&](size_t c) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double v, a;
                if (c < scales.size()) {
                    v = scales[c].first * v_cur[d];
                    a = scales[c].second * a_cur[d];
                } else {
                    v = fractions[c - scales.size()] * tangent_speed[k] * unit_tangent[idx(k, d)];
                    a = 0.0;
                }
                clip_state(k, d, v, a);
                vel[idx(k, d)] = v;
                acc[idx(k, d)] = a;
            }
        };

        std::optional<size_t> best_candidate;
        for (size_t c = 0; c < scales.size() + fractions.size(); ++c) {
            set_candidate(c);
            double cost, duration;
            if (!evaluate_smoothing(input, k, delta_time, cost, duration) || duration > duration_limit) {
                continue;
            }
            if (accept_smoothing_candidate(cost, duration, best_cost, duration_current)) {
                best_cost = cost;
                best_duration = duration;
                best_candidate = c;
            }
        }

        if (!best_candidate) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                vel[idx(k, d)] = v_cur[d];
                acc[idx(k, d)] = a_cur[d];
            }
            return false;
        }
        set_candidate(*best_candidate);

        duration_budget = std::max(0.0, duration_budget - (best_duration - duration_current));
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!enabled[d]) continue;
            double t_prev, t_next;
            if (evaluate(k, d, vel[idx(k, d)], acc[idx(k, d)], block_prev, block_next, t_prev, t_next)) {
                blocks[idx(k - 1, d)] = block_prev;
                blocks[idx(k, d)] = block_next;
                durations[k - 1] = t_prev;
                durations[k] = t_next;
            }
        }
        return true;
    }

    //! Evaluate deviation and duration of the sections k_first..k_last
    bool evaluate_smoothing_range(const InputParameter<DOFs, CustomVector>& input, size_t k_first, size_t k_last, double delta_time, double& cost, double& duration) {
        cost = 0.0;
        duration = 0.0;
        for (size_t s = k_first; s <= k_last; ++s) {
            if (!compute_section(input, s, delta_time)) {
                return false;
            }
            duration += section_trajectory.get_duration();
            cost += section_deviation_cost(s);
        }
        return true;
    }

    //! Reduce the path deviation of section k by changing the states of DoF d at both of its (intermediate) waypoints together
    bool smooth_section_dof(const InputParameter<DOFs, CustomVector>& input, size_t k, size_t d, double delta_time, double& duration_budget) {
        const size_t i0 = idx(k, d), i1 = idx(k + 1, d);
        const double v0 = vel[i0], a0 = acc[i0], v1 = vel[i1], a1 = acc[i1];

        double best_cost, best_duration;
        if (!evaluate_smoothing_range(input, k - 1, k + 1, delta_time, best_cost, best_duration)) {
            return false;
        }
        const double duration_current = best_duration;
        const double duration_limit = best_duration + duration_budget + duration_eps;

        constexpr std::array<double, 4> scales {{0.75, 0.5, 0.25, 0.0}};
        constexpr std::array<double, 4> fractions {{0.25, 0.5, 0.75, 1.0}};
        auto set_candidate = [&](size_t c) {
            double cv0, ca0, cv1, ca1;
            if (c < scales.size()) {
                cv0 = scales[c] * v0; ca0 = scales[c] * a0;
                cv1 = scales[c] * v1; ca1 = scales[c] * a1;
            } else {
                const double f = fractions[c - scales.size()];
                cv0 = f * tangent_speed[k] * unit_tangent[i0]; ca0 = 0.0;
                cv1 = f * tangent_speed[k + 1] * unit_tangent[i1]; ca1 = 0.0;
            }
            clip_state(k, d, cv0, ca0);
            clip_state(k + 1, d, cv1, ca1);
            vel[i0] = cv0; acc[i0] = ca0;
            vel[i1] = cv1; acc[i1] = ca1;
        };

        std::optional<size_t> best_candidate;
        for (size_t c = 0; c < scales.size() + fractions.size(); ++c) {
            set_candidate(c);
            double cost, duration;
            if (!evaluate_smoothing_range(input, k - 1, k + 1, delta_time, cost, duration) || duration > duration_limit) {
                continue;
            }
            if (accept_smoothing_candidate(cost, duration, best_cost, duration_current)) {
                best_cost = cost;
                best_duration = duration;
                best_candidate = c;
            }
        }

        if (!best_candidate) {
            vel[i0] = v0; acc[i0] = a0;
            vel[i1] = v1; acc[i1] = a1;
            return false;
        }
        set_candidate(*best_candidate);
        duration_budget = std::max(0.0, duration_budget - (best_duration - duration_current));

        for (const size_t kk: {k, k + 1}) {
            double t_prev, t_next;
            if (evaluate(kk, d, vel[idx(kk, d)], acc[idx(kk, d)], block_prev, block_next, t_prev, t_next)) {
                blocks[idx(kk - 1, d)] = block_prev;
                blocks[idx(kk, d)] = block_next;
                durations[kk - 1] = t_prev;
                durations[kk] = t_next;
            }
        }
        return true;
    }

    //! Coordinated move at waypoint k: a pattern step of the limiting DoF combined with a scaling of the states of all other DoFs
    bool smooth_waypoint_coordinated(const InputParameter<DOFs, CustomVector>& input, size_t k, double delta_time, double step_fraction, double& duration_budget) {
        std::vector<double> v_cur(degrees_of_freedom), a_cur(degrees_of_freedom);
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            v_cur[d] = vel[idx(k, d)];
            a_cur[d] = acc[idx(k, d)];
        }

        double best_cost, best_duration;
        if (!evaluate_smoothing(input, k, delta_time, best_cost, best_duration)) {
            return false;
        }
        const double duration_current = best_duration;
        const double duration_limit = best_duration + duration_budget + duration_eps;

        // Limiting DoFs of the adjacent sections
        std::vector<size_t> leaders;
        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (enabled[d] && (blocks[idx(k - 1, d)].t_min > durations[k - 1] - 1e-6 || blocks[idx(k, d)].t_min > durations[k] - 1e-6)) {
                leaders.push_back(d);
            }
        }
        if (leaders.empty()) {
            return false;
        }

        constexpr std::array<double, 4> scales {{0.75, 0.5, 0.25, 0.0}};
        std::optional<std::tuple<size_t, double, double, double>> best;  // leader, v, a, scale
        auto set_candidate = [&](size_t leader, double v_leader, double a_leader, double scale) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double v = (d == leader) ? v_leader : scale * v_cur[d];
                double a = (d == leader) ? a_leader : scale * a_cur[d];
                clip_state(k, d, v, a);
                vel[idx(k, d)] = v;
                acc[idx(k, d)] = a;
            }
        };

        for (const size_t leader: leaders) {
            const double dv = step_fraction * std::max(v_upper(k, leader) - v_lower(k, leader), state_eps);
            const double da = step_fraction * std::max(a_upper(k, leader) - a_lower(k, leader), state_eps);
            const std::array<std::pair<double, double>, 9> moves {{{0.0, 0.0}, {dv, 0.0}, {-dv, 0.0}, {0.0, da}, {0.0, -da}, {dv, da}, {dv, -da}, {-dv, da}, {-dv, -da}}};
            for (const auto& [mv, ma]: moves) {
                for (const double scale: scales) {
                    set_candidate(leader, v_cur[leader] + mv, a_cur[leader] + ma, scale);
                    double cost, duration;
                    if (!evaluate_smoothing(input, k, delta_time, cost, duration) || duration > duration_limit) {
                        continue;
                    }
                    if (accept_smoothing_candidate(cost, duration, best_cost, duration_current)) {
                        best_cost = cost;
                        best_duration = duration;
                        best = {leader, v_cur[leader] + mv, a_cur[leader] + ma, scale};
                    }
                }
            }
        }

        if (!best) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                vel[idx(k, d)] = v_cur[d];
                acc[idx(k, d)] = a_cur[d];
            }
            return false;
        }
        set_candidate(std::get<0>(*best), std::get<1>(*best), std::get<2>(*best), std::get<3>(*best));
        duration_budget = std::max(0.0, duration_budget - (best_duration - duration_current));

        for (size_t d = 0; d < degrees_of_freedom; ++d) {
            if (!enabled[d]) continue;
            double t_prev, t_next;
            if (evaluate(k, d, vel[idx(k, d)], acc[idx(k, d)], block_prev, block_next, t_prev, t_next)) {
                blocks[idx(k - 1, d)] = block_prev;
                blocks[idx(k, d)] = block_next;
                durations[k - 1] = t_prev;
                durations[k] = t_next;
            }
        }
        return true;
    }

    void smoothing(const InputParameter<DOFs, CustomVector>& input, double delta_time, double duration_budget) {
        duration_budget = std::max(0.0, duration_budget);
        double step_fraction {0.25};
        for (size_t step = 0; step < number_smoothing_steps; ++step) {
            bool improved {false};
            for (size_t k = 1; k < n_sections; ++k) {
                improved |= smooth_waypoint_coordinated(input, k, delta_time, step_fraction, duration_budget);
            }
            for (size_t k = 1; k + 1 < n_sections; ++k) {
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (enabled[d]) {
                        improved |= smooth_section_dof(input, k, d, delta_time, duration_budget);
                    }
                }
            }
            for (size_t k = 1; k < n_sections; ++k) {
                improved |= smooth_waypoint_joint(input, k, delta_time, duration_budget);
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (enabled[d]) {
                        improved |= smooth_waypoint_dof(input, k, d, delta_time, step_fraction, duration_budget);
                    }
                }
            }
            if (check_deadline()) {
                return;
            }
            if (!improved) {
                step_fraction *= 0.5;
                if (step_fraction < 1e-3) {
                    return;
                }
            }
        }
    }

    //! Assemble the final trajectory from the waypoint states
    template<bool throw_error>
    Result assemble(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& traj, double delta_time) {
        traj.degrees_of_freedom = degrees_of_freedom;
        traj.resize(n_sections - 1);
        traj.continue_calculation_counter = 0;

        for (size_t attempt = 0; attempt < 64; ++attempt) {
            double time {0.0};
            bool success {true};
            std::vector<double> disabled_position(degrees_of_freedom), disabled_velocity(degrees_of_freedom), disabled_acceleration(degrees_of_freedom);
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                disabled_position[d] = input.current_position[d];
                disabled_velocity[d] = input.current_velocity[d];
                disabled_acceleration[d] = input.current_acceleration[d];
            }

            for (size_t k = 0; k < n_sections; ++k) {
                fill_section_input(input, k);
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (!enabled[d]) {
                        section_input.current_position[d] = disabled_position[d];
                        section_input.current_velocity[d] = disabled_velocity[d];
                        section_input.current_acceleration[d] = disabled_acceleration[d];
                    }
                }

                bool interrupted {false};
                const Result result = section_calculator.template calculate<false>(section_input, section_trajectory, delta_time, interrupted);
                if (result != Result::Working) {
                    // Numerical issue in this section: fall back to a slower state at the following (or the last intermediate) waypoint
                    const size_t k_fallback = std::min(k + 1, n_sections - 1);
                    for (size_t d = 0; d < degrees_of_freedom; ++d) {
                        vel[idx(k_fallback, d)] *= 0.5;
                        acc[idx(k_fallback, d)] *= 0.5;
                        if (std::abs(vel[idx(k_fallback, d)]) < 1e-3 * std::max(limits[idx(k, d)].v_max, state_eps)) vel[idx(k_fallback, d)] = 0.0;
                        if (std::abs(acc[idx(k_fallback, d)]) < 1e-3 * std::max(limits[idx(k, d)].a_max, state_eps)) acc[idx(k_fallback, d)] = 0.0;
                    }
                    success = false;
                    break;
                }

                // Time-synchronized profiles with a reduced jerk might exceed the velocity limit within their first or last phases
                // (an unchecked corner case of the state-to-state calculation). Back off the responsible waypoint state in that case.
                bool velocity_ok {true};
                for (size_t d = 0; d < degrees_of_freedom && velocity_ok; ++d) {
                    if (!enabled[d]) continue;
                    const Profile& p = section_trajectory.profiles[0][d];
                    const Limits& l = limits[idx(k, d)];
                    for (size_t i = 0; i < 7; ++i) {
                        if (p.t[i] <= 0.0 || p.j[i] == 0.0 || p.a[i] * p.a[i + 1] >= 0.0) continue;
                        const double v_extremum = p.v[i] - p.a[i] * p.a[i] / (2 * p.j[i]);
                        if (v_extremum > l.v_max + 1e-9 || v_extremum < l.v_min - 1e-9) {
                            const size_t k_state = (i < 3) ? k : k + 1;
                            if (k_state >= 1 && k_state < n_sections) {
                                acc[idx(k_state, d)] *= 0.5;
                                if (std::abs(acc[idx(k_state, d)]) < 1e-3 * std::max(l.a_max, state_eps)) {
                                    acc[idx(k_state, d)] = 0.0;
                                    vel[idx(k_state, d)] *= 0.9;
                                }
                                velocity_ok = false;
                            }
                            break;
                        }
                    }
                }
                if (!velocity_ok) {
                    success = false;
                    break;
                }

                const double section_duration = section_trajectory.get_duration();
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    traj.profiles[k][d] = section_trajectory.profiles[0][d];
                    if (!enabled[d]) {
                        std::tie(disabled_position[d], disabled_velocity[d], disabled_acceleration[d]) = integrate(section_duration, disabled_position[d], disabled_velocity[d], disabled_acceleration[d], 0.0);
                    }
                }
                time += section_duration;
                traj.cumulative_times[k] = time;
            }

            if (success) {
                traj.duration = time;
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    double independent {0.0};
                    for (size_t k = 0; k < n_sections; ++k) {
                        independent += blocks[idx(k, d)].t_min;
                    }
                    traj.independent_min_durations[d] = independent;
                }
                return Result::Working;
            }
        }

        if constexpr (throw_error) {
            throw RuckigError("error in waypoint section calculation, input: " + input.to_string());
        }
        return Result::ErrorSynchronizationCalculation;
    }

    //! Check the (optional) positional limits of the calculated trajectory
    bool check_positional_limits(const InputParameter<DOFs, CustomVector>& input, const Trajectory<DOFs, CustomVector>& traj) {
        constexpr double p_eps {1e-9};
        for (size_t k = 0; k < n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                if (!enabled[d]) {
                    continue;
                }
                const double p_max = input.per_section_max_position ? input.per_section_max_position.value()[k][d] : input.max_position[d];
                const double p_min = input.per_section_min_position ? input.per_section_min_position.value()[k][d] : input.min_position[d];
                if (std::isinf(p_max) && std::isinf(p_min)) {
                    continue;
                }
                const Bound extrema = traj.profiles[k][d].get_position_extrema();
                if (extrema.max > p_max + p_eps || extrema.min < p_min - p_eps) {
                    return false;
                }
            }
        }
        return true;
    }

    void initialize_section_input() {
        section_input.min_velocity = section_input.max_velocity;
        section_input.min_acceleration = section_input.max_acceleration;
        if constexpr (DOFs == 0) {
            sample_position.resize(degrees_of_freedom);
            sample_velocity.resize(degrees_of_freedom);
            sample_acceleration.resize(degrees_of_freedom);
            position_extrema.resize(degrees_of_freedom);
        }
    }

public:
    size_t degrees_of_freedom;

    //! Number of random perturbations (each followed by a short local descent) to escape local minima
    size_t number_global_steps {32};

    //! Number of sweeps of the coordinate-wise pattern search over all waypoint states
    size_t number_local_steps {32};

    //! Number of sweeps for reducing the path deviation without increasing the trajectory duration
    size_t number_smoothing_steps {8};

    //! Allowed relative increase of the trajectory duration during the smoothing steps (in favor of a lower path deviation)
    double smoothing_duration_tolerance {0.0};

    //! Optional warm start: velocities and accelerations at the intermediate waypoints, indexed by [waypoint][dof],
    //! e.g. from a previous calculation with the same waypoints. Ignored if the sizes do not match the input.
    std::vector<std::vector<double>> initial_velocities, initial_accelerations;

    //! Print information about the optimization progress to stdout
    bool debug {false};


    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator(): degrees_of_freedom(DOFs) {
        initialize_section_input();
    }

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t): degrees_of_freedom(DOFs) {
        initialize_section_input();
    }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs):
        section_calculator(TargetCalculator<DOFs, CustomVector>(dofs)),
        section_input(InputParameter<DOFs, CustomVector>(dofs)),
        section_trajectory(Trajectory<DOFs, CustomVector>(dofs)),
        degrees_of_freedom(dofs)
    {
        initialize_section_input();
    }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs, size_t): LocalWaypointsCalculator(dofs) { }

    //! Calculate a trajectory along the intermediate waypoints of the input
    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& traj, double delta_time, bool& was_interrupted) {
        const auto start = Clock::now();
        was_interrupted = false;

        if (input.intermediate_positions.empty()) {
            return section_calculator.template calculate<throw_error>(input, traj, delta_time, was_interrupted);
        }

        deadline = std::nullopt;
        deadline_reached = false;
        if (input.interrupt_calculation_duration) {
            deadline = start + std::chrono::microseconds(static_cast<long long>(input.interrupt_calculation_duration.value()));
        }

        setup(input);
        solutions.clear();

        // 1. Structured initializations
        set_zero_states();
        double best_total = recompute_all();
        if (std::isinf(best_total)) {
            if constexpr (throw_error) {
                throw RuckigError("error in calculating the section durations of the waypoint trajectory, input: " + input.to_string());
            }
            return Result::ErrorExecutionTimeCalculation;
        }
        save_best();

        if (debug) std::cout << "[ruckig] waypoints: stop at every waypoint: " << best_total << std::endl;

        auto try_current = [&](const char* label) {
            const double total = recompute_all();
            if (std::isinf(total)) {
                return;
            }
            local_descent(number_local_steps, 0.25, Objective::Joint);
            const double result = total_duration();
            if (debug) std::cout << "[ruckig] waypoints: " << label << ": " << total << " -> " << result << std::endl;
            remember_solution();
            if (result < best_total - duration_eps) {
                best_total = result;
                save_best();
            }
        };

        // Stop at every waypoint
        try_current("descent from stop");

        // Warm start from given waypoint states
        if (initial_velocities.size() == n_sections - 1 && initial_accelerations.size() == n_sections - 1 && !check_deadline()) {
            bool valid {true};
            for (size_t k = 1; k < n_sections && valid; ++k) {
                if (initial_velocities[k - 1].size() != degrees_of_freedom || initial_accelerations[k - 1].size() != degrees_of_freedom) {
                    valid = false;
                    break;
                }
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    double v = initial_velocities[k - 1][d], a = initial_accelerations[k - 1][d];
                    clip_state(k, d, v, a);
                    vel[idx(k, d)] = v;
                    acc[idx(k, d)] = a;
                }
            }
            if (valid && !std::isinf(recompute_all())) {
                remember_solution();
                if (total_duration() < best_total - duration_eps) {
                    best_total = total_duration();
                    save_best();
                }
                try_current("descent from initial states");
            }
        }

        // Follow the path tangent at various speeds
        for (const double scale: {0.5, 1.0}) {
            if (check_deadline()) break;
            set_tangent_states(scale);
            try_current(scale < 0.75 ? "descent from tangent 0.5" : "descent from tangent 1.0");
        }

        // Time-optimal traversal of every DoF on its own: the slowest DoF leads, the others follow
        independent_vel.assign(vel.size(), 0.0);
        independent_acc.assign(acc.size(), 0.0);
        if (!check_deadline()) {
            set_tangent_states(0.5);
            if (!std::isinf(recompute_all())) {
                local_descent(number_local_steps, 0.25, Objective::Independent);
                independent_vel = vel;
                independent_acc = acc;

                std::vector<std::pair<double, size_t>> independent_durations(degrees_of_freedom);
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    double sum {0.0};
                    for (size_t k = 0; k < n_sections; ++k) {
                        sum += blocks[idx(k, d)].t_min;
                    }
                    independent_durations[d] = {sum, d};
                }
                std::sort(independent_durations.rbegin(), independent_durations.rend());
                if (debug) {
                    std::cout << "[ruckig] waypoints: independent optimum per DoF:";
                    for (const auto& [sum, d]: independent_durations) std::cout << " dof " << d << ": " << sum;
                    std::cout << std::endl;
                }

                // Try the slowest DoFs as leader with the others at rest or following the tangent
                const size_t number_leaders = std::min<size_t>(degrees_of_freedom, 2);
                for (size_t l = 0; l < number_leaders; ++l) {
                    const size_t leader = independent_durations[l].second;
                    if (!enabled[leader]) continue;
                    for (const double scale: {0.0, 0.5}) {
                        if (check_deadline()) break;
                        set_tangent_states(scale);
                        for (size_t k = 1; k < n_sections; ++k) {
                            vel[idx(k, leader)] = independent_vel[idx(k, leader)];
                            acc[idx(k, leader)] = independent_acc[idx(k, leader)];
                        }
                        try_current(scale < 0.25 ? "descent from leader (others at rest)" : "descent from leader (others tangent 0.5)");
                    }
                }
            }
        }

        if (debug) std::cout << "[ruckig] waypoints: duration after initializations: " << best_total << " (" << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " ms)" << std::endl;

        // 2. Global steps: re-initialize the states of a waypoint (all DoFs), of a single DoF at a waypoint, or of two consecutive
        // waypoints, and repair the neighborhood by a local descent. The kicks are ordered by their typical benefit.
        struct Kick { size_t k, width, type; std::optional<size_t> dof; };
        const size_t n_waypoints = n_sections - 1;
        std::vector<Kick> kicks;
        for (const size_t type: {3, 0}) {
            for (size_t k = 1; k <= n_waypoints; ++k) kicks.push_back({k, 1, type, std::nullopt});
        }
        for (const size_t type: {3, 0}) {
            for (size_t k = 1; k <= n_waypoints; ++k) {
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (enabled[d]) kicks.push_back({k, 1, type, d});
                }
            }
        }
        for (const size_t type: {2, 1}) {
            for (size_t k = 1; k <= n_waypoints; ++k) kicks.push_back({k, 1, type, std::nullopt});
        }
        for (size_t width = 2; width <= n_waypoints; ++width) {
            for (const size_t type: {3, 0}) {
                for (size_t k = 1; k + width - 1 <= n_waypoints; ++k) kicks.push_back({k, width, type, std::nullopt});
            }
        }

        for (size_t step = 0; step < std::min(number_global_steps, kicks.size()); ++step) {
            if (check_deadline()) break;

            const Kick& kick = kicks[step];
            restore_best();
            for (size_t kk = kick.k; kk < kick.k + kick.width; ++kk) {
                for (size_t d = 0; d < degrees_of_freedom; ++d) {
                    if (!enabled[d] || (kick.dof && *kick.dof != d)) continue;
                    double v {0.0}, a {0.0};
                    switch (kick.type) {
                        case 0: break;
                        case 1: v = 0.5 * v_tangent[idx(kk, d)]; break;
                        case 2: v = v_tangent[idx(kk, d)]; break;
                        default: v = independent_vel[idx(kk, d)]; a = independent_acc[idx(kk, d)]; break;
                    }
                    clip_state(kk, d, v, a);
                    vel[idx(kk, d)] = v;
                    acc[idx(kk, d)] = a;
                }
            }
            if (std::isinf(recompute_all())) {
                continue;
            }
            local_descent(std::max<size_t>(number_local_steps / 4, 2), 0.125, Objective::Joint, kick.k - 1, kick.k + kick.width);
            local_descent(std::max<size_t>(number_local_steps / 8, 2), 0.03125, Objective::Joint);
            const double result = total_duration();
            remember_solution();
            if (result < best_total - duration_eps) {
                if (debug) std::cout << "[ruckig] waypoints: global step " << step << " (waypoint " << kick.k << ", type " << kick.type << ", width " << kick.width << (kick.dof ? ", dof " + std::to_string(*kick.dof) : std::string()) << "): " << best_total << " -> " << result << std::endl;
                best_total = result;
                save_best();
            }
        }

        restore_best();
        recompute_all();
        if (debug) std::cout << "[ruckig] waypoints: duration after global steps: " << total_duration() << " (" << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " ms)" << std::endl;

        // 3. Among all local optima within the duration tolerance, select the one with the lowest path deviation.
        // A longer duration is only accepted for a substantial reduction of the maximum deviation.
        double duration_budget = std::max(0.0, smoothing_duration_tolerance) * best_total;
        if (duration_budget > 0.0 && !check_deadline()) {
            double fastest_max, fastest_mean;
            if (solution_deviation(input, delta_time, fastest_max, fastest_mean)) {
                constexpr double required_improvement {0.1};
                double best_max {fastest_max * (1.0 - required_improvement)}, best_mean {fastest_mean};
                const Solution* selected {nullptr};
                for (const auto& s: solutions) {
                    if (s.duration > best_total + duration_budget + duration_eps || s.duration < best_total + duration_eps) {
                        continue;
                    }
                    vel = s.vel;
                    acc = s.acc;
                    double max_deviation, mean_deviation;
                    if (!solution_deviation(input, delta_time, max_deviation, mean_deviation)) {
                        continue;
                    }
                    if (max_deviation < best_max - 1e-9 || (max_deviation < best_max + 1e-9 && mean_deviation < best_mean)) {
                        best_max = max_deviation;
                        best_mean = mean_deviation;
                        selected = &s;
                    }
                }
                if (selected) {
                    vel = selected->vel;
                    acc = selected->acc;
                    duration_budget -= selected->duration - best_total;
                    if (debug) std::cout << "[ruckig] waypoints: selected solution with duration " << selected->duration << " and deviation " << best_max << " (mean " << best_mean << ") among " << solutions.size() << " instead of " << fastest_max << std::endl;
                } else {
                    restore_best();
                }
            } else {
                restore_best();
            }
            recompute_all();
        }

        // 4. Smoothing: reduce the path deviation within the remaining duration budget
        if (number_smoothing_steps > 0 && !check_deadline()) {
            smoothing(input, delta_time, duration_budget);
            if (debug) std::cout << "[ruckig] waypoints: duration after smoothing: " << total_duration() << " (" << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " ms)" << std::endl;
        }

        was_interrupted = deadline_reached;

        // 5. Final trajectory
        const Result result = assemble<throw_error>(input, traj, delta_time);
        if (result != Result::Working) {
            return result;
        }

        if (!check_positional_limits(input, traj)) {
            return Result::ErrorPositionalLimits;
        }
        return Result::Working;
    }

    //! Calculate the trajectory for given velocities and accelerations at the intermediate waypoints (without optimization).
    //! The states are indexed by [waypoint][dof] and are clipped to the kinematic limits.
    template<bool throw_error>
    Result calculate_from_states(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& traj, double delta_time, const std::vector<std::vector<double>>& velocities, const std::vector<std::vector<double>>& accelerations) {
        bool was_interrupted {false};
        if (input.intermediate_positions.empty()) {
            return section_calculator.template calculate<throw_error>(input, traj, delta_time, was_interrupted);
        }
        if (velocities.size() != input.intermediate_positions.size() || accelerations.size() != input.intermediate_positions.size()) {
            if constexpr (throw_error) {
                throw RuckigError("number of waypoint states does not match the number of intermediate positions.");
            }
            return Result::ErrorInvalidInput;
        }

        deadline = std::nullopt;
        deadline_reached = false;
        setup(input);
        for (size_t k = 1; k < n_sections; ++k) {
            for (size_t d = 0; d < degrees_of_freedom; ++d) {
                double v = velocities[k - 1][d], a = accelerations[k - 1][d];
                clip_state(k, d, v, a);
                vel[idx(k, d)] = v;
                acc[idx(k, d)] = a;
            }
        }
        recompute_all();

        const Result result = assemble<throw_error>(input, traj, delta_time);
        if (result != Result::Working) {
            return result;
        }
        return check_positional_limits(input, traj) ? Result::Working : Result::ErrorPositionalLimits;
    }

    //! Continue the trajectory calculation (not available for the local waypoints calculator)
    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>&, Trajectory<DOFs, CustomVector>&, double, bool&) {
        if constexpr (throw_error) {
            throw RuckigError("continue calculation is not available for the local waypoints calculator.");
        }
        return Result::Error;
    }
};

} // namespace ruckig
