#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include <ruckig/calculator_target.hpp>
#include <ruckig/error.hpp>
#include <ruckig/input_parameter.hpp>
#include <ruckig/result.hpp>
#include <ruckig/trajectory.hpp>
#include <ruckig/utils.hpp>


namespace ruckig {

//! Calculation class for waypoint trajectories based on Kiemel & Kroeger 2024.
template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
class LocalWaypointsCalculator {
    template<class T> using Vector = CustomVector<T, DOFs>;

    struct ScalarLimits {
        double max_velocity;
        double min_velocity;
        double max_acceleration;
        double min_acceleration;
        double max_jerk;
    };

    struct ScalarPlan {
        bool valid {false};
        Profile profile;
        double duration {0.0};
    };

    struct ScalarPath {
        std::vector<size_t> point_indices;
        std::vector<double> positions;
        std::vector<double> accelerations;
        std::vector<ScalarLimits> section_limits;
        std::vector<ScalarPlan> fastest_plans;
        std::vector<ScalarPlan> timed_plans;
        std::vector<double> cumulative_times;
        double fastest_duration {0.0};
        double duration {0.0};
    };

    constexpr static double eps {1e-12};
    constexpr static double binary_precision {1e-8};
    constexpr static size_t binary_iterations {48};

    size_t degrees_of_freedom;

    template<typename T>
    static double get_section_limit(const std::optional<std::vector<Vector<double>>>& per_section, const T& global, size_t section, size_t dof) {
        if (per_section && section < per_section->size()) {
            return per_section.value()[section][dof];
        }
        return global[dof];
    }

    ScalarLimits limits_for_section(const InputParameter<DOFs, CustomVector>& input, size_t section, size_t dof) const {
        const double max_velocity = get_section_limit(input.per_section_max_velocity, input.max_velocity, section, dof);
        const double min_velocity = input.per_section_min_velocity
            ? get_section_limit(input.per_section_min_velocity, input.min_velocity.value_or(input.max_velocity), section, dof)
            : (input.min_velocity ? input.min_velocity.value()[dof] : -max_velocity);

        const double max_acceleration = get_section_limit(input.per_section_max_acceleration, input.max_acceleration, section, dof);
        const double min_acceleration = input.per_section_min_acceleration
            ? get_section_limit(input.per_section_min_acceleration, input.min_acceleration.value_or(input.max_acceleration), section, dof)
            : (input.min_acceleration ? input.min_acceleration.value()[dof] : -max_acceleration);

        return {
            max_velocity,
            min_velocity,
            max_acceleration,
            min_acceleration,
            get_section_limit(input.per_section_max_jerk, input.max_jerk, section, dof),
        };
    }

    ScalarLimits limits_for_range(const InputParameter<DOFs, CustomVector>& input, size_t first_point, size_t last_point, size_t dof) const {
        auto limits = limits_for_section(input, first_point, dof);
        for (size_t section = first_point + 1; section < last_point; ++section) {
            const auto section_limits = limits_for_section(input, section, dof);
            limits.max_velocity = std::min(limits.max_velocity, section_limits.max_velocity);
            limits.min_velocity = std::max(limits.min_velocity, section_limits.min_velocity);
            limits.max_acceleration = std::min(limits.max_acceleration, section_limits.max_acceleration);
            limits.min_acceleration = std::max(limits.min_acceleration, section_limits.min_acceleration);
            limits.max_jerk = std::min(limits.max_jerk, section_limits.max_jerk);
        }
        return limits;
    }

    static Profile make_hold_profile(double position, double duration) {
        Profile profile;
        profile.t = {0.0, 0.0, 0.0, duration, 0.0, 0.0, 0.0};
        profile.t_sum = {0.0, 0.0, 0.0, duration, duration, duration, duration};
        profile.j = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        profile.a = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        profile.v = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        profile.p = {position, position, position, position, position, position, position, position};
        profile.pf = position;
        profile.vf = 0.0;
        profile.af = 0.0;
        profile.brake.duration = 0.0;
        profile.accel.duration = 0.0;
        profile.limits = Profile::ReachedLimits::NONE;
        profile.direction = Profile::Direction::UP;
        profile.control_signs = Profile::ControlSigns::UDDU;
        return profile;
    }

    static ScalarPlan plan_scalar(double p0, double v0, double a0, double pf, double vf, double af, const ScalarLimits& limits, std::optional<double> minimum_duration = std::nullopt) {
        if (std::abs(pf - p0) < eps && std::abs(v0) < eps && std::abs(vf) < eps && std::abs(a0) < eps && std::abs(af) < eps) {
            const double duration = minimum_duration.value_or(0.0);
            return {true, make_hold_profile(p0, duration), duration};
        }

        InputParameter<1> scalar_input;
        scalar_input.current_position = {p0};
        scalar_input.current_velocity = {v0};
        scalar_input.current_acceleration = {a0};
        scalar_input.target_position = {pf};
        scalar_input.target_velocity = {vf};
        scalar_input.target_acceleration = {af};
        scalar_input.max_velocity = {limits.max_velocity};
        scalar_input.min_velocity = std::array<double, 1>{limits.min_velocity};
        scalar_input.max_acceleration = {limits.max_acceleration};
        scalar_input.min_acceleration = std::array<double, 1>{limits.min_acceleration};
        scalar_input.max_jerk = {limits.max_jerk};
        if (minimum_duration) {
            scalar_input.minimum_duration = minimum_duration;
        }

        TargetCalculator<1> scalar_calculator;
        Trajectory<1> scalar_trajectory;
        bool was_interrupted {false};
        const auto result = scalar_calculator.template calculate<false>(scalar_input, scalar_trajectory, 0.0, was_interrupted);
        if (result != Result::Working) {
            return {};
        }

        const auto profiles = scalar_trajectory.get_profiles();
        return {true, profiles[0][0], scalar_trajectory.get_duration()};
    }

    static bool is_monotonic_profile(const Profile& profile, double p0, double pf) {
        const double delta = pf - p0;
        if (std::abs(delta) < eps) {
            return true;
        }

        const double direction = (delta > 0.0) ? 1.0 : -1.0;
        const double lower = std::min(p0, pf) - 1e-7;
        const double upper = std::max(p0, pf) + 1e-7;
        const double duration = profile.t_sum.back() + profile.brake.duration + profile.accel.duration;
        const size_t samples = 64;

        for (size_t i = 0; i <= samples; ++i) {
            const double t = duration * static_cast<double>(i) / static_cast<double>(samples);
            size_t segment = 0;
            double local_t = t;
            while (segment < profile.t_sum.size() && local_t > profile.t_sum[segment]) {
                ++segment;
            }

            if (segment >= profile.t.size()) {
                segment = profile.t.size() - 1;
                local_t = profile.t[segment];
            } else if (segment > 0) {
                local_t -= profile.t_sum[segment - 1];
            }

            const auto state = integrate(local_t, profile.p[segment], profile.v[segment], profile.a[segment], profile.j[segment]);
            const double position = std::get<0>(state);
            const double velocity = std::get<1>(state);
            if (position < lower || position > upper) {
                return false;
            }
            if (std::abs(velocity) > 1e-7 && direction * velocity < -1e-7) {
                return false;
            }
        }

        return true;
    }

    static double acceleration_limit_in_direction(double sign, const ScalarLimits& limits) {
        return sign >= 0.0 ? limits.max_acceleration : -limits.min_acceleration;
    }

    static double section_direction(double from, double to) {
        const double delta = to - from;
        if (std::abs(delta) < eps) {
            return 0.0;
        }
        return delta > 0.0 ? 1.0 : -1.0;
    }

    static double direction_before(const std::vector<std::vector<double>>& points, size_t point, size_t dof) {
        for (size_t i = point; i > 0; --i) {
            const double direction = section_direction(points[i - 1][dof], points[i][dof]);
            if (direction != 0.0) {
                return direction;
            }
        }
        return 0.0;
    }

    static double direction_after(const std::vector<std::vector<double>>& points, size_t point, size_t dof) {
        for (size_t i = point; i + 1 < points.size(); ++i) {
            const double direction = section_direction(points[i][dof], points[i + 1][dof]);
            if (direction != 0.0) {
                return direction;
            }
        }
        return 0.0;
    }

    static std::tuple<double, double, double> state_at_profile_time(const Profile& profile, double time) {
        if (time <= 0.0) {
            if (profile.brake.duration > 0.0) {
                return {profile.brake.p[0], profile.brake.v[0], profile.brake.a[0]};
            }
            return {profile.p[0], profile.v[0], profile.a[0]};
        }

        if (profile.brake.duration > 0.0) {
            if (time < profile.brake.duration) {
                if (time < profile.brake.t[0] || profile.brake.t[1] <= 0.0) {
                    return integrate(time, profile.brake.p[0], profile.brake.v[0], profile.brake.a[0], profile.brake.j[0]);
                }

                return integrate(time - profile.brake.t[0], profile.brake.p[1], profile.brake.v[1], profile.brake.a[1], profile.brake.j[1]);
            }
            time -= profile.brake.duration;
        }

        if (time >= profile.t_sum.back()) {
            return {profile.p.back(), profile.v.back(), profile.a.back()};
        }

        size_t segment = 0;
        while (segment + 1 < profile.t_sum.size() && time > profile.t_sum[segment]) {
            ++segment;
        }

        double local_time = time;
        if (segment > 0) {
            local_time -= profile.t_sum[segment - 1];
        }
        return integrate(local_time, profile.p[segment], profile.v[segment], profile.a[segment], profile.j[segment]);
    }

    static Profile slice_profile(const Profile& profile, double offset) {
        const double source_duration = profile.t_sum.back();
        if (offset >= source_duration - eps) {
            return make_hold_profile(profile.p.back(), 0.0);
        }

        size_t source_segment = 0;
        while (source_segment + 1 < profile.t_sum.size() && offset > profile.t_sum[source_segment]) {
            ++source_segment;
        }

        double local_offset = offset;
        if (source_segment > 0) {
            local_offset -= profile.t_sum[source_segment - 1];
        }

        auto state = state_at_profile_time(profile, offset);
        double position = std::get<0>(state);
        double velocity = std::get<1>(state);
        double acceleration = std::get<2>(state);

        Profile sliced;
        sliced.t = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        sliced.t_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        sliced.j = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        sliced.a = {acceleration, acceleration, acceleration, acceleration, acceleration, acceleration, acceleration, acceleration};
        sliced.v = {velocity, velocity, velocity, velocity, velocity, velocity, velocity, velocity};
        sliced.p = {position, position, position, position, position, position, position, position};
        sliced.pf = profile.pf;
        sliced.vf = profile.vf;
        sliced.af = profile.af;
        sliced.brake.duration = 0.0;
        sliced.accel.duration = 0.0;
        sliced.limits = profile.limits;
        sliced.direction = profile.direction;
        sliced.control_signs = profile.control_signs;

        double cumulative = 0.0;
        size_t target_segment = 0;
        for (size_t source = source_segment; source < profile.t.size() && target_segment < sliced.t.size(); ++source) {
            double duration = profile.t[source];
            if (source == source_segment) {
                duration -= local_offset;
            }
            if (duration <= eps) {
                continue;
            }

            sliced.t[target_segment] = duration;
            sliced.j[target_segment] = profile.j[source];
            sliced.p[target_segment] = position;
            sliced.v[target_segment] = velocity;
            sliced.a[target_segment] = acceleration;

            std::tie(position, velocity, acceleration) = integrate(duration, position, velocity, acceleration, sliced.j[target_segment]);
            cumulative += duration;
            sliced.t_sum[target_segment] = cumulative;
            ++target_segment;
        }

        for (size_t i = target_segment; i < sliced.t.size(); ++i) {
            sliced.t_sum[i] = cumulative;
            sliced.p[i] = position;
            sliced.v[i] = velocity;
            sliced.a[i] = acceleration;
        }
        sliced.p.back() = position;
        sliced.v.back() = velocity;
        sliced.a.back() = acceleration;

        return sliced;
    }

    static Profile truncate_profile(const Profile& profile, double duration) {
        Profile truncated;
        double position = profile.brake.duration > 0.0 ? profile.brake.p[0] : profile.p[0];
        double velocity = profile.brake.duration > 0.0 ? profile.brake.v[0] : profile.v[0];
        double acceleration = profile.brake.duration > 0.0 ? profile.brake.a[0] : profile.a[0];

        truncated.t = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        truncated.t_sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        truncated.j = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        truncated.a = {acceleration, acceleration, acceleration, acceleration, acceleration, acceleration, acceleration, acceleration};
        truncated.v = {velocity, velocity, velocity, velocity, velocity, velocity, velocity, velocity};
        truncated.p = {position, position, position, position, position, position, position, position};
        truncated.pf = profile.pf;
        truncated.vf = profile.vf;
        truncated.af = profile.af;
        truncated.brake.duration = 0.0;
        truncated.accel.duration = 0.0;
        truncated.limits = profile.limits;
        truncated.direction = profile.direction;
        truncated.control_signs = profile.control_signs;

        double remaining = duration;
        double cumulative = 0.0;
        size_t target_segment = 0;

        const auto append_phase = [&](double phase_duration, double jerk) {
            if (remaining <= eps || phase_duration <= eps || target_segment >= truncated.t.size()) {
                return;
            }

            const double use_duration = std::min(phase_duration, remaining);
            truncated.t[target_segment] = use_duration;
            truncated.j[target_segment] = jerk;
            truncated.p[target_segment] = position;
            truncated.v[target_segment] = velocity;
            truncated.a[target_segment] = acceleration;

            std::tie(position, velocity, acceleration) = integrate(use_duration, position, velocity, acceleration, jerk);
            cumulative += use_duration;
            remaining -= use_duration;
            truncated.t_sum[target_segment] = cumulative;
            ++target_segment;
        };

        if (profile.brake.duration > 0.0) {
            append_phase(profile.brake.t[0], profile.brake.j[0]);
            append_phase(profile.brake.t[1], profile.brake.j[1]);
        }

        for (size_t source = 0; source < profile.t.size(); ++source) {
            append_phase(profile.t[source], profile.j[source]);
        }

        if (remaining > eps && target_segment < truncated.t.size()) {
            append_phase(remaining, 0.0);
        }

        for (size_t i = target_segment; i < truncated.t.size(); ++i) {
            truncated.t_sum[i] = cumulative;
            truncated.p[i] = position;
            truncated.v[i] = velocity;
            truncated.a[i] = acceleration;
        }
        truncated.p.back() = position;
        truncated.v.back() = velocity;
        truncated.a.back() = acceleration;
        truncated.pf = position;
        truncated.vf = velocity;
        truncated.af = acceleration;

        return truncated;
    }

    static double search_target_acceleration(double p0, double pf, double target_sign, const ScalarLimits& limits) {
        if (target_sign == 0.0) {
            return 0.0;
        }

        double lower = 0.0;
        double upper = acceleration_limit_in_direction(target_sign, limits);
        for (size_t i = 0; i < binary_iterations; ++i) {
            const double candidate = 0.5 * (lower + upper);
            const auto plan = plan_scalar(p0, 0.0, 0.0, pf, 0.0, target_sign * candidate, limits);
            if (plan.valid && is_monotonic_profile(plan.profile, p0, pf)) {
                lower = candidate;
            } else {
                upper = candidate;
            }
            if (upper - lower < binary_precision) {
                break;
            }
        }
        return lower;
    }

    static double search_input_acceleration(double p0, double pf, double input_sign, const ScalarLimits& limits) {
        if (input_sign == 0.0) {
            return 0.0;
        }

        double lower = 0.0;
        double upper = acceleration_limit_in_direction(input_sign, limits);
        for (size_t i = 0; i < binary_iterations; ++i) {
            const double candidate = 0.5 * (lower + upper);
            const auto plan = plan_scalar(p0, 0.0, input_sign * candidate, pf, 0.0, 0.0, limits);
            if (plan.valid && is_monotonic_profile(plan.profile, p0, pf)) {
                lower = candidate;
            } else {
                upper = candidate;
            }
            if (upper - lower < binary_precision) {
                break;
            }
        }
        return lower;
    }

    std::vector<std::vector<double>> collect_points(const InputParameter<DOFs, CustomVector>& input) const {
        std::vector<std::vector<double>> points(input.intermediate_positions.size() + 2, std::vector<double>(degrees_of_freedom));
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            points[0][dof] = input.current_position[dof];
            for (size_t i = 0; i < input.intermediate_positions.size(); ++i) {
                points[i + 1][dof] = input.intermediate_positions[i][dof];
            }
            points.back()[dof] = input.target_position[dof];
        }
        return points;
    }

    ScalarPath build_fastest_scalar_path(const InputParameter<DOFs, CustomVector>& input, const std::vector<std::vector<double>>& points, size_t dof) const {
        ScalarPath path;
        path.point_indices.push_back(0);
        for (size_t point = 1; point + 1 < points.size(); ++point) {
            const double before = direction_before(points, point, dof);
            const double after = direction_after(points, point, dof);
            if (before != 0.0 && after != 0.0 && before != after) {
                path.point_indices.push_back(point);
            }
        }
        if (path.point_indices.back() != points.size() - 1) {
            path.point_indices.push_back(points.size() - 1);
        }

        const size_t point_count = path.point_indices.size();
        path.positions.resize(point_count);
        path.accelerations.assign(point_count, 0.0);
        for (size_t i = 0; i < point_count; ++i) {
            path.positions[i] = points[path.point_indices[i]][dof];
        }

        if (point_count == 1) {
            return path;
        }

        path.section_limits.reserve(point_count - 1);
        for (size_t section = 0; section + 1 < point_count; ++section) {
            path.section_limits.push_back(limits_for_range(input, path.point_indices[section], path.point_indices[section + 1], dof));
        }

        path.accelerations.front() = input.current_acceleration[dof];
        path.accelerations.back() = input.target_acceleration[dof];
        for (size_t point = 1; point + 1 < point_count; ++point) {
            const double next_direction = section_direction(path.positions[point], path.positions[point + 1]);
            if (next_direction == 0.0) {
                continue;
            }

            const double a_out_max = search_target_acceleration(path.positions[point - 1], path.positions[point], next_direction, path.section_limits[point - 1]);
            const double a_in_max = search_input_acceleration(path.positions[point], path.positions[point + 1], next_direction, path.section_limits[point]);
            path.accelerations[point] = next_direction * std::min(a_out_max, a_in_max);
        }

        path.fastest_plans.resize(point_count - 1);
        path.cumulative_times.resize(point_count - 1, 0.0);
        for (size_t section = 0; section + 1 < point_count; ++section) {
            const double v0 = (section == 0) ? input.current_velocity[dof] : 0.0;
            const double vf = (section + 2 == point_count) ? input.target_velocity[dof] : 0.0;
            path.fastest_plans[section] = plan_scalar(
                path.positions[section], v0, path.accelerations[section],
                path.positions[section + 1], vf, path.accelerations[section + 1],
                path.section_limits[section]
            );
            if (!path.fastest_plans[section].valid) {
                return path;
            }

            path.fastest_duration += path.fastest_plans[section].duration;
            path.cumulative_times[section] = path.fastest_duration;
        }
        path.duration = path.fastest_duration;
        return path;
    }

    bool time_scale_scalar_path(const InputParameter<DOFs, CustomVector>& input, ScalarPath& path, size_t dof, double target_duration) const {
        if (path.fastest_plans.empty()) {
            path.timed_plans = {ScalarPlan{true, make_hold_profile(path.positions.empty() ? input.current_position[dof] : path.positions.front(), target_duration), target_duration}};
            path.cumulative_times = {target_duration};
            path.duration = target_duration;
            return true;
        }

        const double scale = path.fastest_duration > eps ? target_duration / path.fastest_duration : 1.0;
        path.timed_plans.resize(path.fastest_plans.size());
        path.cumulative_times.assign(path.fastest_plans.size(), 0.0);
        path.duration = 0.0;

        for (size_t section = 0; section < path.fastest_plans.size(); ++section) {
            const double v0 = (section == 0) ? input.current_velocity[dof] : 0.0;
            const double vf = (section + 1 == path.fastest_plans.size()) ? input.target_velocity[dof] : 0.0;
            const double section_duration = std::max(path.fastest_plans[section].duration, path.fastest_plans[section].duration * scale);
            auto timed_plan = plan_scalar(
                path.positions[section], v0, path.accelerations[section],
                path.positions[section + 1], vf, path.accelerations[section + 1],
                path.section_limits[section],
                section_duration
            );
            if (!timed_plan.valid) {
                if (std::abs(section_duration - path.fastest_plans[section].duration) < 1e-9) {
                    timed_plan = path.fastest_plans[section];
                } else {
                    return false;
                }
            }

            path.timed_plans[section] = timed_plan;
            path.duration += timed_plan.duration;
            path.cumulative_times[section] = path.duration;
        }

        if (path.duration < target_duration - 1e-9) {
            path.timed_plans.push_back({true, make_hold_profile(path.positions.back(), target_duration - path.duration), target_duration - path.duration});
            path.duration = target_duration;
            path.cumulative_times.push_back(target_duration);
        }

        return true;
    }

    static void add_unique_time(std::vector<double>& times, double time) {
        if (time < eps) {
            return;
        }
        for (double existing: times) {
            if (std::abs(existing - time) < 1e-9) {
                return;
            }
        }
        times.push_back(time);
    }

    static size_t profile_index_at_time(const ScalarPath& path, double time) {
        auto it = std::upper_bound(path.cumulative_times.begin(), path.cumulative_times.end(), time + 1e-10);
        size_t index = static_cast<size_t>(std::distance(path.cumulative_times.begin(), it));
        if (index >= path.timed_plans.size()) {
            index = path.timed_plans.size() - 1;
        }
        return index;
    }

    static std::vector<double> path_arc_lengths(const std::vector<std::vector<double>>& points) {
        std::vector<double> arc(points.size(), 0.0);
        for (size_t i = 1; i < points.size(); ++i) {
            double squared_length = 0.0;
            for (size_t dof = 0; dof < points[i].size(); ++dof) {
                const double delta = points[i][dof] - points[i - 1][dof];
                squared_length += delta * delta;
            }
            arc[i] = arc[i - 1] + std::sqrt(squared_length);
        }
        return arc;
    }

    static size_t arc_segment_at(const std::vector<double>& arc, double path_length) {
        auto it = std::upper_bound(arc.begin(), arc.end(), path_length + 1e-12);
        if (it == arc.begin()) {
            return 0;
        }
        size_t section = static_cast<size_t>(std::distance(arc.begin(), it) - 1);
        if (section + 1 >= arc.size()) {
            section = arc.size() - 2;
        }
        return section;
    }

    static double position_at_path_length(const std::vector<std::vector<double>>& points, const std::vector<double>& arc, size_t dof, double path_length) {
        if (path_length <= 0.0) {
            return points.front()[dof];
        }
        if (path_length >= arc.back()) {
            return points.back()[dof];
        }

        const size_t section = arc_segment_at(arc, path_length);
        const double section_length = arc[section + 1] - arc[section];
        const double alpha = section_length > eps ? (path_length - arc[section]) / section_length : 0.0;
        return points[section][dof] + alpha * (points[section + 1][dof] - points[section][dof]);
    }

    static double path_length_from_scalar_state(const ScalarPath& path, const std::vector<std::vector<double>>& points, const std::vector<double>& arc, size_t scalar_section, size_t dof, double position) {
        if (path.point_indices.size() < 2 || scalar_section + 1 >= path.point_indices.size()) {
            return arc.back();
        }

        const size_t path_start = path.point_indices[scalar_section];
        const size_t path_end = path.point_indices[scalar_section + 1];

        for (size_t point = path_start; point < path_end; ++point) {
            const double p0 = points[point][dof];
            const double p1 = points[point + 1][dof];
            if (std::abs(p1 - p0) < eps) {
                continue;
            }

            const double lower = std::min(p0, p1) - 1e-9;
            const double upper = std::max(p0, p1) + 1e-9;
            if (position < lower || position > upper) {
                continue;
            }

            const double alpha = std::clamp((position - p0) / (p1 - p0), 0.0, 1.0);
            return arc[point] + alpha * (arc[point + 1] - arc[point]);
        }

        const double start_distance = std::abs(position - points[path_start][dof]);
        const double end_distance = std::abs(position - points[path_end][dof]);
        return start_distance <= end_distance ? arc[path_start] : arc[path_end];
    }

    static double reference_path_length_at_time(const ScalarPath& path, const std::vector<std::vector<double>>& points, const std::vector<double>& arc, size_t dof, double time) {
        if (time >= path.duration - eps) {
            return arc.back();
        }

        const size_t profile_index = profile_index_at_time(path, time);
        const double profile_start = (profile_index == 0) ? 0.0 : path.cumulative_times[profile_index - 1];
        const double offset = std::max(0.0, time - profile_start);
        const auto state = state_at_profile_time(path.timed_plans[profile_index].profile, offset);
        return path_length_from_scalar_state(path, points, arc, profile_index, dof, std::get<0>(state));
    }

    bool assemble_tracking_trajectory(
        const InputParameter<DOFs, CustomVector>& input,
        const std::vector<std::vector<double>>& points,
        const std::vector<ScalarPath>& scalar_paths,
        double step_size,
        Trajectory<DOFs, CustomVector>& trajectory
    ) const {
        if (scalar_paths.empty()) {
            return false;
        }

        const auto arc = path_arc_lengths(points);
        if (arc.back() < eps) {
            trajectory.resize(0);
            trajectory.duration = 0.0;
            trajectory.cumulative_times[0] = 0.0;
            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                trajectory.profiles[0][dof] = make_hold_profile(input.current_position[dof], 0.0);
                trajectory.independent_min_durations[dof] = 0.0;
            }
            return true;
        }

        size_t reference_dof = 0;
        for (size_t dof = 1; dof < degrees_of_freedom; ++dof) {
            if (scalar_paths[dof].fastest_duration > scalar_paths[reference_dof].fastest_duration) {
                reference_dof = dof;
            }
        }

        const double duration = scalar_paths[reference_dof].duration;
        if (duration < eps) {
            return false;
        }

        step_size = std::clamp(step_size, 1e-4, 0.01);
        const size_t steps = std::max<size_t>(1, static_cast<size_t>(std::ceil(duration / step_size)));
        std::vector<double> times(steps + 1, 0.0);
        for (size_t i = 1; i <= steps; ++i) {
            times[i] = std::min(duration, static_cast<double>(i) * step_size);
        }

        std::vector<double> reference_u(steps + 1, 0.0);
        for (size_t i = 0; i <= steps; ++i) {
            reference_u[i] = reference_path_length_at_time(scalar_paths[reference_dof], points, arc, reference_dof, times[i]);
        }
        reference_u.front() = 0.0;
        reference_u.back() = arc.back();

        std::vector<std::vector<double>> desired_position(steps + 1, std::vector<double>(degrees_of_freedom, 0.0));
        std::vector<std::vector<double>> desired_velocity(steps + 1, std::vector<double>(degrees_of_freedom, 0.0));
        std::vector<ScalarLimits> global_limits;
        global_limits.reserve(degrees_of_freedom);
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            global_limits.push_back(limits_for_range(input, 0, points.size() - 1, dof));
        }

        for (size_t i = 0; i <= steps; ++i) {
            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                desired_position[i][dof] = position_at_path_length(points, arc, dof, reference_u[i]);
                desired_velocity[i][dof] = 0.0;
            }
        }

        desired_position.front() = points.front();
        desired_position.back() = points.back();
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            desired_velocity.front()[dof] = input.current_velocity[dof];
            desired_velocity.back()[dof] = input.target_velocity[dof];
        }

        std::vector<std::vector<Profile>> generated_profiles;
        generated_profiles.reserve(steps + 1);
        std::vector<double> generated_durations;
        generated_durations.reserve(steps + 1);
        std::vector<size_t> generated_public_sections;
        generated_public_sections.reserve(steps + 1);

        std::vector<double> current_position = points.front();
        std::vector<double> current_velocity(degrees_of_freedom, 0.0);
        std::vector<double> current_acceleration(degrees_of_freedom, 0.0);
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            current_velocity[dof] = input.current_velocity[dof];
            current_acceleration[dof] = input.current_acceleration[dof];
        }

        for (size_t step = 0; step < steps; ++step) {
            const double dt = times[step + 1] - times[step];
            std::vector<Profile> section_profiles(degrees_of_freedom);

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                auto plan = plan_scalar(
                    current_position[dof], current_velocity[dof], current_acceleration[dof],
                    desired_position[step + 1][dof], desired_velocity[step + 1][dof], 0.0,
                    global_limits[dof],
                    dt
                );
                if (!plan.valid) {
                    plan = plan_scalar(
                        current_position[dof], current_velocity[dof], current_acceleration[dof],
                        desired_position[step + 1][dof], 0.0, 0.0,
                        global_limits[dof],
                        dt
                    );
                }
                if (!plan.valid) {
                    return false;
                }

                section_profiles[dof] = truncate_profile(plan.profile, dt);
                std::tie(current_position[dof], current_velocity[dof], current_acceleration[dof]) = state_at_profile_time(section_profiles[dof], dt);
            }

            generated_profiles.push_back(section_profiles);
            generated_durations.push_back(dt);
            generated_public_sections.push_back(arc_segment_at(arc, reference_u[step]));
        }

        bool needs_final_profile = false;
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            needs_final_profile = needs_final_profile
                || std::abs(current_position[dof] - input.target_position[dof]) > 1e-8
                || std::abs(current_velocity[dof] - input.target_velocity[dof]) > 1e-8
                || std::abs(current_acceleration[dof] - input.target_acceleration[dof]) > 1e-8;
        }

        if (needs_final_profile) {
            std::vector<Profile> final_profiles(degrees_of_freedom);
            double final_duration = 0.0;
            std::vector<ScalarPlan> final_plans(degrees_of_freedom);
            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                final_plans[dof] = plan_scalar(
                    current_position[dof], current_velocity[dof], current_acceleration[dof],
                    input.target_position[dof], input.target_velocity[dof], input.target_acceleration[dof],
                    global_limits[dof]
                );
                if (!final_plans[dof].valid) {
                    return false;
                }
                final_duration = std::max(final_duration, final_plans[dof].duration);
            }

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                auto plan = plan_scalar(
                    current_position[dof], current_velocity[dof], current_acceleration[dof],
                    input.target_position[dof], input.target_velocity[dof], input.target_acceleration[dof],
                    global_limits[dof],
                    final_duration
                );
                if (!plan.valid) {
                    plan = final_plans[dof];
                }
                if (plan.profile.brake.duration > 0.0) {
                    return false;
                }
                final_profiles[dof] = plan.profile;
            }
            generated_profiles.push_back(final_profiles);
            generated_durations.push_back(final_duration);
            generated_public_sections.push_back(points.size() - 2);
        }

        trajectory.resize(generated_profiles.size() - 1);
        double cumulative_time = 0.0;
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            trajectory.independent_min_durations[dof] = scalar_paths[dof].fastest_duration;
        }
        for (size_t section = 0; section < generated_profiles.size(); ++section) {
            cumulative_time += generated_durations[section];
            trajectory.cumulative_times[section] = cumulative_time;
            trajectory.public_sections[section] = generated_public_sections[section];
            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                trajectory.profiles[section][dof] = generated_profiles[section][dof];
            }
        }
        trajectory.duration = cumulative_time;
        return true;
    }

public:
    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t = 0): degrees_of_freedom(DOFs) { }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs): degrees_of_freedom(dofs) { }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit LocalWaypointsCalculator(size_t dofs, size_t): degrees_of_freedom(dofs) { }

    //! Calculate a jerk-limited local waypoint trajectory.
    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
        was_interrupted = false;

        if (input.per_section_minimum_duration && input.per_section_minimum_duration->size() != input.intermediate_positions.size() + 1) {
            if constexpr (throw_error) {
                throw RuckigError("per_section_minimum_duration must have one entry per local waypoint section.");
            }
            return Result::ErrorInvalidInput;
        }

        const auto points = collect_points(input);
        const size_t section_count = points.size() - 1;
        if (section_count == 0) {
            return Result::ErrorInvalidInput;
        }

        std::vector<ScalarPath> scalar_paths(degrees_of_freedom);
        double target_duration = input.minimum_duration.value_or(0.0);
        if (input.per_section_minimum_duration) {
            for (double duration: input.per_section_minimum_duration.value()) {
                target_duration += duration;
            }
        }

        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            scalar_paths[dof] = build_fastest_scalar_path(input, points, dof);
            if (scalar_paths[dof].fastest_plans.empty()) {
                scalar_paths[dof].positions = {input.current_position[dof]};
            }
            for (const auto& plan: scalar_paths[dof].fastest_plans) {
                if (!plan.valid) {
                    if constexpr (throw_error) {
                        throw RuckigError("local waypoint backend failed to compute one-dimensional extrema traversal for dof " + std::to_string(dof) + ".");
                    }
                    return Result::ErrorExecutionTimeCalculation;
                }
            }
            target_duration = std::max(target_duration, scalar_paths[dof].fastest_duration);
        }

        std::vector<double> global_times {0.0};
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            if (!time_scale_scalar_path(input, scalar_paths[dof], dof, target_duration)) {
                if constexpr (throw_error) {
                    throw RuckigError("local waypoint backend failed to synchronize one-dimensional extrema traversal for dof " + std::to_string(dof) + ".");
                }
                return Result::ErrorSynchronizationCalculation;
            }
            for (double time: scalar_paths[dof].cumulative_times) {
                add_unique_time(global_times, time);
            }
        }

        const double tracking_step = delta_time > 0.0 ? std::min(delta_time, 0.0025) : 0.0025;
        if (degrees_of_freedom > 1 && assemble_tracking_trajectory(input, points, scalar_paths, tracking_step, trajectory)) {
            trajectory.continue_calculation_counter = 0;
            return Result::Working;
        }

        add_unique_time(global_times, target_duration);
        std::sort(global_times.begin(), global_times.end());

        if (global_times.size() == 1) {
            global_times.push_back(0.0);
        }

        const size_t trajectory_section_count = global_times.size() - 1;
        trajectory.resize(trajectory_section_count - 1);
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            trajectory.independent_min_durations[dof] = scalar_paths[dof].fastest_duration;
        }

        const auto fallback_arc = path_arc_lengths(points);
        size_t reference_dof = 0;
        for (size_t dof = 1; dof < degrees_of_freedom; ++dof) {
            if (scalar_paths[dof].fastest_duration > scalar_paths[reference_dof].fastest_duration) {
                reference_dof = dof;
            }
        }

        for (size_t section = 0; section < trajectory_section_count; ++section) {
            const double section_start = global_times[section];
            const double section_end = global_times[section + 1];
            trajectory.cumulative_times[section] = section_end;
            if (fallback_arc.size() > 1 && fallback_arc.back() > eps) {
                const double reference_u = reference_path_length_at_time(scalar_paths[reference_dof], points, fallback_arc, reference_dof, section_start);
                trajectory.public_sections[section] = arc_segment_at(fallback_arc, reference_u);
            }

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                const auto& path = scalar_paths[dof];
                const size_t profile_index = profile_index_at_time(path, section_start);
                const double profile_start = (profile_index == 0) ? 0.0 : path.cumulative_times[profile_index - 1];
                const double offset = std::max(0.0, section_start - profile_start);
                trajectory.profiles[section][dof] = slice_profile(path.timed_plans[profile_index].profile, offset);
            }
        }

        trajectory.duration = global_times.back();
        trajectory.continue_calculation_counter = 0;

        return Result::Working;
    }

    //! Continue the trajectory calculation.
    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
        return calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
    }
};

} // namespace ruckig
