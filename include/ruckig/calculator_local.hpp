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

    std::vector<std::vector<double>> compute_waypoint_accelerations(const InputParameter<DOFs, CustomVector>& input, const std::vector<std::vector<double>>& points) const {
        const size_t point_count = points.size();
        std::vector<std::vector<double>> accelerations(point_count, std::vector<double>(degrees_of_freedom, 0.0));

        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            accelerations.front()[dof] = input.current_acceleration[dof];
            accelerations.back()[dof] = input.target_acceleration[dof];

            for (size_t point = 1; point + 1 < point_count; ++point) {
                const double previous_direction = section_direction(points[point - 1][dof], points[point][dof]);
                const double next_direction = section_direction(points[point][dof], points[point + 1][dof]);
                if (previous_direction == 0.0 || next_direction == 0.0) {
                    accelerations[point][dof] = 0.0;
                    continue;
                }

                const auto outgoing_limits = limits_for_section(input, point - 1, dof);
                const auto incoming_limits = limits_for_section(input, point, dof);
                const double a_out_max = search_target_acceleration(points[point - 1][dof], points[point][dof], next_direction, outgoing_limits);
                const double a_in_max = search_input_acceleration(points[point][dof], points[point + 1][dof], next_direction, incoming_limits);
                accelerations[point][dof] = next_direction * std::min(a_out_max, a_in_max);
            }
        }

        return accelerations;
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
        (void)delta_time;
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

        trajectory.resize(section_count - 1);
        for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
            trajectory.independent_min_durations[dof] = 0.0;
        }

        const auto waypoint_accelerations = compute_waypoint_accelerations(input, points);
        std::vector<double> cumulative_times(section_count, 0.0);
        double cumulative_time = 0.0;

        for (size_t section = 0; section < section_count; ++section) {
            std::vector<ScalarPlan> fastest_plans(degrees_of_freedom);
            double section_duration = input.per_section_minimum_duration ? input.per_section_minimum_duration.value()[section] : 0.0;

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                const auto limits = limits_for_section(input, section, dof);
                const double v0 = (section == 0) ? input.current_velocity[dof] : 0.0;
                const double vf = (section + 1 == section_count) ? input.target_velocity[dof] : 0.0;
                fastest_plans[dof] = plan_scalar(
                    points[section][dof], v0, waypoint_accelerations[section][dof],
                    points[section + 1][dof], vf, waypoint_accelerations[section + 1][dof],
                    limits
                );

                if (!fastest_plans[dof].valid) {
                    if constexpr (throw_error) {
                        throw RuckigError("local waypoint backend failed to compute the upper one-dimensional trajectory for section " + std::to_string(section) + ", dof " + std::to_string(dof) + ".");
                    }
                    return Result::ErrorExecutionTimeCalculation;
                }
                section_duration = std::max(section_duration, fastest_plans[dof].duration);
            }

            for (size_t dof = 0; dof < degrees_of_freedom; ++dof) {
                const auto limits = limits_for_section(input, section, dof);
                const double v0 = (section == 0) ? input.current_velocity[dof] : 0.0;
                const double vf = (section + 1 == section_count) ? input.target_velocity[dof] : 0.0;
                auto timed_plan = plan_scalar(
                    points[section][dof], v0, waypoint_accelerations[section][dof],
                    points[section + 1][dof], vf, waypoint_accelerations[section + 1][dof],
                    limits,
                    section_duration
                );

                if (!timed_plan.valid) {
                    if (std::abs(fastest_plans[dof].duration - section_duration) < 1e-9) {
                        timed_plan = fastest_plans[dof];
                    } else {
                        if constexpr (throw_error) {
                            throw RuckigError("local waypoint backend failed to synchronize section " + std::to_string(section) + ", dof " + std::to_string(dof) + ".");
                        }
                        return Result::ErrorSynchronizationCalculation;
                    }
                }
                trajectory.profiles[section][dof] = timed_plan.profile;
                trajectory.independent_min_durations[dof] = std::max(trajectory.independent_min_durations[dof], fastest_plans[dof].duration);
            }

            cumulative_time += section_duration;
            cumulative_times[section] = cumulative_time;
        }

        trajectory.duration = cumulative_time;
        for (size_t section = 0; section < section_count; ++section) {
            trajectory.cumulative_times[section] = cumulative_times[section];
        }
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
