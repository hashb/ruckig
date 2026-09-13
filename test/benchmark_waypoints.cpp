// Compare the local waypoints calculator with the cloud API (if available) on a set of problems.
//
// Reports for each backend: trajectory duration, maximum/mean deviation from the piecewise-linear
// path through the waypoints, kinematic-limit validity, waypoint accuracy, and calculation time.
//
// Usage: benchmark-waypoints [--no-cloud] [--seed N] [--random N] [--verbose] [--problem NAME]
//                            [--global N] [--local N] [--smooth N] [--tolerance X]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <ruckig/ruckig.hpp>


using namespace ruckig;

using Vec = std::vector<double>;


struct Problem {
    std::string name;
    InputParameter<DynamicDOFs> input;

    explicit Problem(const std::string& name, size_t dofs): name(name), input(dofs) { }
};


struct Report {
    bool ok {false};
    Result result {Result::Error};
    double duration {0.0};
    double max_deviation {0.0};
    double mean_deviation {0.0};
    double max_waypoint_error {0.0};
    double max_velocity_violation {0.0}, max_acceleration_violation {0.0}, max_jerk_violation {0.0};
    double max_discontinuity {0.0};
    double calculation_time_ms {0.0};
    std::vector<double> cumulative_times;
};


static double point_to_segment_distance(const Vec& p, const Vec& a, const Vec& b) {
    double len_sq {0.0}, dot {0.0};
    for (size_t d = 0; d < p.size(); ++d) {
        len_sq += (b[d] - a[d]) * (b[d] - a[d]);
        dot += (p[d] - a[d]) * (b[d] - a[d]);
    }
    const double h = (len_sq > 1e-18) ? std::clamp(dot / len_sq, 0.0, 1.0) : 0.0;
    double dist_sq {0.0};
    for (size_t d = 0; d < p.size(); ++d) {
        const double diff = p[d] - a[d] - h * (b[d] - a[d]);
        dist_sq += diff * diff;
    }
    return std::sqrt(dist_sq);
}


static Report evaluate_trajectory(const Trajectory<DynamicDOFs>& trajectory, const Problem& problem, bool verbose, Report report) {
    const size_t dofs = problem.input.degrees_of_freedom;
    if (report.result != Result::Working && report.result != Result::ErrorPositionalLimits) {
        return report;
    }

    report.duration = trajectory.get_duration();
    report.cumulative_times = trajectory.get_intermediate_durations();

    // Reference polyline
    std::vector<Vec> polyline;
    polyline.push_back(problem.input.current_position);
    for (const auto& wp: problem.input.intermediate_positions) {
        polyline.push_back(wp);
    }
    polyline.push_back(problem.input.target_position);

    // Sample the trajectory
    const double dt = 1e-3;
    const size_t n = static_cast<size_t>(std::ceil(report.duration / dt)) + 1;
    Vec p(dofs), v(dofs), a(dofs), j(dofs);
    Vec p_prev(dofs), v_prev(dofs), a_prev(dofs);
    std::vector<Vec> positions;
    positions.reserve(n);

    double sum_deviation {0.0};
    double t_max_deviation {0.0};
    size_t section_max_deviation {0};
    Vec p_max_deviation(dofs);
    for (size_t i = 0; i < n; ++i) {
        const double t = std::min(i * dt, report.duration);
        size_t section;
        trajectory.at_time(t, p, v, a, j, section);
        positions.push_back(p);

        double deviation = std::numeric_limits<double>::infinity();
        for (size_t s = 0; s + 1 < polyline.size(); ++s) {
            deviation = std::min(deviation, point_to_segment_distance(p, polyline[s], polyline[s + 1]));
        }
        if (deviation > report.max_deviation) {
            report.max_deviation = deviation;
            t_max_deviation = t;
            section_max_deviation = section;
            p_max_deviation = p;
        }
        sum_deviation += deviation;

        for (size_t d = 0; d < dofs; ++d) {
            report.max_velocity_violation = std::max(report.max_velocity_violation, std::abs(v[d]) - problem.input.max_velocity[d]);
            report.max_acceleration_violation = std::max(report.max_acceleration_violation, std::abs(a[d]) - problem.input.max_acceleration[d]);
            report.max_jerk_violation = std::max(report.max_jerk_violation, std::abs(j[d]) - problem.input.max_jerk[d]);

            if (i > 0) {
                // Continuity: compare with the integration of the previous state
                const auto [p_int, v_int, a_int] = integrate(dt, p_prev[d], v_prev[d], a_prev[d], 0.0);
                (void)p_int; (void)v_int;
                report.max_discontinuity = std::max(report.max_discontinuity, std::abs(a[d] - a_int) - problem.input.max_jerk[d] * dt * 1.0001);
            }
        }
        p_prev = p; v_prev = v; a_prev = a;
    }
    report.mean_deviation = sum_deviation / n;

    // Waypoint accuracy
    for (size_t w = 0; w < problem.input.intermediate_positions.size(); ++w) {
        trajectory.at_time(report.cumulative_times[w], p);
        for (size_t d = 0; d < dofs; ++d) {
            report.max_waypoint_error = std::max(report.max_waypoint_error, std::abs(p[d] - problem.input.intermediate_positions[w][d]));
        }
    }
    trajectory.at_time(report.duration, p, v, a);
    for (size_t d = 0; d < dofs; ++d) {
        report.max_waypoint_error = std::max(report.max_waypoint_error, std::abs(p[d] - problem.input.target_position[d]));
        report.max_waypoint_error = std::max(report.max_waypoint_error, std::abs(v[d] - problem.input.target_velocity[d]));
        report.max_waypoint_error = std::max(report.max_waypoint_error, std::abs(a[d] - problem.input.target_acceleration[d]));
    }

    report.ok = report.max_velocity_violation < 1e-6 && report.max_acceleration_violation < 1e-6 && report.max_jerk_violation < 1e-6 && report.max_waypoint_error < 1e-6;

    if (verbose) {
        std::printf("    max deviation %.4f at t=%.3f in section %zu at p=[", report.max_deviation, t_max_deviation, section_max_deviation);
        for (size_t d = 0; d < dofs; ++d) std::printf("%s%+.3f", d ? ", " : "", p_max_deviation[d]);
        std::printf("]  (waypoints [");
        for (size_t d = 0; d < dofs; ++d) std::printf("%s%+.3f", d ? ", " : "", polyline[section_max_deviation][d]);
        std::printf("] -> [");
        for (size_t d = 0; d < dofs; ++d) std::printf("%s%+.3f", d ? ", " : "", polyline[std::min(section_max_deviation + 1, polyline.size() - 1)][d]);
        std::printf("])\n");

        const auto profiles = trajectory.get_profiles();
        Ruckig<1> single(0.01);
        InputParameter<1> single_input;
        Trajectory<1> single_trajectory;

        double t_start {0.0};
        for (size_t s = 0; s < profiles.size(); ++s) {
            const double section_duration = report.cumulative_times[s] - t_start;
            std::printf("    section %zu: duration %.4f\n", s, section_duration);
            for (size_t d = 0; d < dofs; ++d) {
                const auto& pr = profiles[s][d];

                // Independent minimum duration of this DoF for the same boundary states
                single_input.current_position = {pr.p[0]};
                single_input.current_velocity = {pr.v[0]};
                single_input.current_acceleration = {pr.a[0]};
                single_input.target_position = {pr.pf};
                single_input.target_velocity = {pr.vf};
                single_input.target_acceleration = {pr.af};
                single_input.max_velocity = {problem.input.max_velocity[d]};
                single_input.max_acceleration = {problem.input.max_acceleration[d]};
                single_input.max_jerk = {problem.input.max_jerk[d]};
                double t_min {-1.0};
                if (single.calculate(single_input, single_trajectory) == Result::Working) {
                    t_min = single_trajectory.get_duration() + pr.brake.duration;
                }
                const bool limiting = std::abs(t_min - section_duration) < 1e-6;
                std::printf("      dof %zu: p %+.3f -> %+.3f  v0=%+.4f a0=%+.4f -> vf=%+.4f af=%+.4f  t_min=%.4f %s\n", d, pr.p[0], pr.pf, pr.v[0], pr.a[0], pr.vf, pr.af, t_min, limiting ? "<== limiting" : "");
            }
            t_start = report.cumulative_times[s];
        }
    }
    return report;
}


static Report evaluate(Ruckig<DynamicDOFs>& otg, const Problem& problem, bool verbose) {
    Report report;
    const size_t dofs = problem.input.degrees_of_freedom;
    Trajectory<DynamicDOFs> trajectory(dofs, problem.input.intermediate_positions.size());

    const auto start = std::chrono::steady_clock::now();
    report.result = otg.calculate(problem.input, trajectory);
    const auto stop = std::chrono::steady_clock::now();
    report.calculation_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return evaluate_trajectory(trajectory, problem, verbose, report);
}


static std::vector<Problem> make_problems(size_t n_random, int seed) {
    std::vector<Problem> problems;

    {
        Problem p("example-03", 3);
        p.input.current_position = {0.2, 0.0, -0.3};
        p.input.current_velocity = {0.0, 0.2, 0.0};
        p.input.current_acceleration = {0.0, 0.6, 0.0};
        p.input.intermediate_positions = {{1.4, -1.6, 1.0}, {-0.6, -0.5, 0.4}, {-0.4, -0.35, 0.0}, {0.8, 1.8, -0.1}};
        p.input.target_position = {0.5, 1.0, 0.0};
        p.input.target_velocity = {0.2, 0.0, 0.3};
        p.input.target_acceleration = {0.0, 0.1, -0.1};
        p.input.max_velocity = {1.0, 2.0, 1.0};
        p.input.max_acceleration = {3.0, 2.0, 2.0};
        p.input.max_jerk = {6.0, 10.0, 20.0};
        problems.push_back(p);
    }
    {
        Problem p("example-08", 3);
        p.input.current_position = {0.8, 0.0, 0.5};
        p.input.intermediate_positions = {{1.4, -1.6, 1.0}, {-0.6, -0.5, 0.4}, {-0.4, -0.35, 0.0}, {-0.2, 0.35, -0.1}, {0.2, 0.5, -0.1}, {0.8, 1.8, -0.1}};
        p.input.target_position = {0.5, 1.2, 0.0};
        p.input.max_velocity = {3.0, 2.0, 2.0};
        p.input.max_acceleration = {6.0, 4.0, 4.0};
        p.input.max_jerk = {16.0, 10.0, 20.0};
        problems.push_back(p);
    }
    {
        Problem p("single-dof", 1);
        p.input.current_position = {0.0};
        p.input.intermediate_positions = {{1.0}, {2.0}, {1.5}, {3.0}};
        p.input.target_position = {2.0};
        p.input.max_velocity = {1.0};
        p.input.max_acceleration = {2.0};
        p.input.max_jerk = {10.0};
        problems.push_back(p);
    }
    {
        Problem p("straight-line", 2);
        p.input.current_position = {0.0, 0.0};
        p.input.intermediate_positions = {{0.5, 0.5}, {1.0, 1.0}, {1.5, 1.5}};
        p.input.target_position = {2.0, 2.0};
        p.input.max_velocity = {1.0, 1.0};
        p.input.max_acceleration = {2.0, 2.0};
        p.input.max_jerk = {10.0, 10.0};
        problems.push_back(p);
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dofs_dist(2, 6);
    std::uniform_int_distribution<size_t> waypoints_dist(1, 6);
    std::normal_distribution<double> position_dist(0.0, 1.0);
    std::uniform_real_distribution<double> vel_limit_dist(0.5, 2.0);
    std::uniform_real_distribution<double> acc_limit_dist(1.0, 5.0);
    std::uniform_real_distribution<double> jerk_limit_dist(5.0, 30.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (size_t i = 0; i < n_random; ++i) {
        const size_t dofs = dofs_dist(rng);
        const size_t n_waypoints = waypoints_dist(rng);
        Problem p("random-" + std::to_string(i), dofs);
        for (size_t d = 0; d < dofs; ++d) {
            p.input.current_position[d] = position_dist(rng);
            p.input.target_position[d] = position_dist(rng);
            p.input.max_velocity[d] = vel_limit_dist(rng);
            p.input.max_acceleration[d] = acc_limit_dist(rng);
            p.input.max_jerk[d] = jerk_limit_dist(rng);
            if (unit(rng) < 0.3) {
                p.input.current_velocity[d] = 0.5 * p.input.max_velocity[d] * (2 * unit(rng) - 1);
                p.input.current_acceleration[d] = 0.5 * p.input.max_acceleration[d] * (2 * unit(rng) - 1);
            }
            if (unit(rng) < 0.3) {
                p.input.target_velocity[d] = 0.5 * p.input.max_velocity[d] * (2 * unit(rng) - 1);
                p.input.target_acceleration[d] = 0.5 * p.input.max_acceleration[d] * (2 * unit(rng) - 1);
            }
        }
        for (size_t w = 0; w < n_waypoints; ++w) {
            std::vector<double> wp(dofs);
            for (size_t d = 0; d < dofs; ++d) {
                wp[d] = position_dist(rng);
            }
            p.input.intermediate_positions.push_back(wp);
        }
        problems.push_back(p);
    }
    return problems;
}


int main(int argc, char** argv) {
    bool use_cloud {true}, verbose {false};
    int seed {42};
    size_t n_random {8};
    std::string filter;
    std::optional<size_t> global_steps, local_steps, smoothing_steps;
    std::optional<double> tolerance;
    bool cloud_states {false};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-cloud") == 0) use_cloud = false;
        else if (std::strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--random") == 0 && i + 1 < argc) n_random = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--problem") == 0 && i + 1 < argc) filter = argv[++i];
        else if (std::strcmp(argv[i], "--global") == 0 && i + 1 < argc) global_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--local") == 0 && i + 1 < argc) local_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--smooth") == 0 && i + 1 < argc) smoothing_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) tolerance = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cloud-states") == 0) cloud_states = true;
    }
#if !defined WITH_CLOUD_CLIENT
    use_cloud = false;
#endif

    const auto problems = make_problems(n_random, seed);

    std::printf("%-14s %8s %8s %8s | %8s %8s %8s | %8s %8s | %s\n", "problem", "local[s]", "cloud[s]", "gap[%]", "loc.max", "cld.max", "loc.mean", "loc[ms]", "cld[ms]", "status");
    double sum_gap {0.0};
    size_t n_gap {0}, n_faster {0}, n_slower {0}, n_fail {0};
    for (const auto& problem: problems) {
        if (!filter.empty() && problem.name.find(filter) == std::string::npos) {
            continue;
        }

        const size_t dofs = problem.input.degrees_of_freedom;
        Ruckig<DynamicDOFs> otg(dofs, 0.01, problem.input.intermediate_positions.size());
        if (global_steps) otg.calculator.local_waypoints_calculator.number_global_steps = *global_steps;
        if (local_steps) otg.calculator.local_waypoints_calculator.number_local_steps = *local_steps;
        if (smoothing_steps) otg.calculator.local_waypoints_calculator.number_smoothing_steps = *smoothing_steps;
        otg.calculator.local_waypoints_calculator.debug = verbose;
        if (tolerance) otg.calculator.local_waypoints_calculator.smoothing_duration_tolerance = *tolerance;

        otg.set_waypoints_backend(WaypointsBackend::Local);
        if (verbose) std::printf("  %s local:\n", problem.name.c_str());
        const Report local = evaluate(otg, problem, verbose);

        Report cloud;
#if defined WITH_CLOUD_CLIENT
        if (use_cloud) {
            otg.set_waypoints_backend(WaypointsBackend::Cloud);
            if (verbose) std::printf("  %s cloud:\n", problem.name.c_str());
            cloud = evaluate(otg, problem, verbose);
        }
#endif

#if defined WITH_CLOUD_CLIENT
        if (use_cloud && cloud_states && cloud.result == Result::Working) {
            // Assemble the cloud's intermediate states with the local calculator and report the result
            Trajectory<DynamicDOFs> cloud_trajectory(dofs, problem.input.intermediate_positions.size());
            otg.set_waypoints_backend(WaypointsBackend::Cloud);
            otg.calculate(problem.input, cloud_trajectory);
            const auto profiles = cloud_trajectory.get_profiles();
            std::vector<std::vector<double>> velocities, accelerations;
            for (size_t s = 0; s + 1 < profiles.size(); ++s) {
                std::vector<double> v(dofs), a(dofs);
                for (size_t d = 0; d < dofs; ++d) {
                    v[d] = profiles[s][d].vf;
                    a[d] = profiles[s][d].af;
                }
                velocities.push_back(v);
                accelerations.push_back(a);
            }
            Trajectory<DynamicDOFs> assembled(dofs, problem.input.intermediate_positions.size());
            Report assembled_report;
            assembled_report.result = otg.calculator.local_waypoints_calculator.template calculate_from_states<false>(problem.input, assembled, 0.01, velocities, accelerations);
            assembled_report = evaluate_trajectory(assembled, problem, false, assembled_report);
            std::printf("  cloud states assembled locally: result %d, duration %.4f (cloud %.4f), max deviation %.4f (cloud %.4f), mean %.4f\n", static_cast<int>(assembled_report.result), assembled_report.duration, cloud.duration, assembled_report.max_deviation, cloud.max_deviation, assembled_report.mean_deviation);

            otg.set_waypoints_backend(WaypointsBackend::Local);
            otg.calculator.local_waypoints_calculator.initial_velocities = velocities;
            otg.calculator.local_waypoints_calculator.initial_accelerations = accelerations;
            const Report warm = evaluate(otg, problem, verbose);
            otg.calculator.local_waypoints_calculator.initial_velocities.clear();
            otg.calculator.local_waypoints_calculator.initial_accelerations.clear();
            std::printf("  local with warm start from cloud states: duration %.4f, max deviation %.4f, mean %.4f, %.1f ms\n", warm.duration, warm.max_deviation, warm.mean_deviation, warm.calculation_time_ms);
        }
#endif

        std::string status;
        if (local.result != Result::Working) {
            status += "LOCAL-ERROR(" + std::to_string(static_cast<int>(local.result)) + ") ";
            n_fail++;
        } else if (!local.ok) {
            status += "LOCAL-INVALID(v" + std::to_string(local.max_velocity_violation) + " a" + std::to_string(local.max_acceleration_violation) + " j" + std::to_string(local.max_jerk_violation) + " wp" + std::to_string(local.max_waypoint_error) + ") ";
            n_fail++;
        }
        if (use_cloud && cloud.result != Result::Working) {
            status += "CLOUD-ERROR ";
        } else if (use_cloud && !cloud.ok) {
            status += "CLOUD-INVALID ";
        }

        double gap {std::numeric_limits<double>::quiet_NaN()};
        if (local.result == Result::Working && cloud.result == Result::Working) {
            gap = (local.duration - cloud.duration) / cloud.duration * 100.0;
            sum_gap += gap;
            n_gap++;
            if (local.duration < cloud.duration * (1.0 - 1e-5)) n_faster++;
            if (local.duration > cloud.duration * (1.0 + 1e-5)) n_slower++;
        }

        std::printf("%-14s %8.4f %8.4f %8.2f | %8.4f %8.4f %8.4f | %8.1f %8.1f | %s\n", problem.name.c_str(), local.duration, cloud.duration, gap, local.max_deviation, cloud.max_deviation, local.mean_deviation, local.calculation_time_ms, cloud.calculation_time_ms, status.c_str());
    }

    if (n_gap > 0) {
        std::printf("\nmean duration gap: %.2f%% over %zu problems (local faster: %zu, slower: %zu), failures: %zu\n", sum_gap / n_gap, n_gap, n_faster, n_slower, n_fail);
    } else {
        std::printf("\nfailures: %zu\n", n_fail);
    }
    return n_fail > 0 ? 1 : 0;
}
