#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <ruckig/ruckig.hpp>


using namespace ruckig;


//! Sample the trajectory finely and check kinematic limits, waypoint accuracy, continuity, and the target state
template<size_t DOFs>
void check_waypoint_trajectory(const InputParameter<DOFs>& input, const Trajectory<DOFs>& trajectory, double tolerance = 1e-6) {
    const size_t dofs = input.degrees_of_freedom;
    const double duration = trajectory.get_duration();
    CHECK( duration >= 0.0 );

    const auto cumulative_times = trajectory.get_intermediate_durations();
    REQUIRE( cumulative_times.size() == input.intermediate_positions.size() + 1 );
    CHECK( cumulative_times.back() == doctest::Approx(duration) );
    for (size_t i = 1; i < cumulative_times.size(); ++i) {
        CHECK( cumulative_times[i] >= cumulative_times[i - 1] - 1e-12 );
    }

    auto make = [&]() {
        if constexpr (DOFs == 0) {
            return std::vector<double>(dofs);
        } else {
            return std::array<double, DOFs>();
        }
    };
    auto p = make(), v = make(), a = make(), j = make();
    auto p_prev = make(), v_prev = make(), a_prev = make();

    const double dt = 5e-4;
    const size_t n = static_cast<size_t>(std::ceil(duration / dt)) + 1;
    for (size_t i = 0; i < n; ++i) {
        const double t = std::min(i * dt, duration);
        size_t section;
        trajectory.at_time(t, p, v, a, j, section);

        for (size_t d = 0; d < dofs; ++d) {
            CHECK( std::abs(v[d]) <= input.max_velocity[d] + tolerance );
            CHECK( std::abs(a[d]) <= input.max_acceleration[d] + tolerance );
            CHECK( std::abs(j[d]) <= input.max_jerk[d] + tolerance );

            if (i > 0) {
                // Continuity of position, velocity, and acceleration between samples
                CHECK( std::abs(p[d] - p_prev[d]) <= (input.max_velocity[d] + tolerance) * dt + tolerance );
                CHECK( std::abs(v[d] - v_prev[d]) <= (input.max_acceleration[d] + tolerance) * dt + tolerance );
                CHECK( std::abs(a[d] - a_prev[d]) <= (input.max_jerk[d] + tolerance) * dt + tolerance );
            }
        }
        p_prev = p; v_prev = v; a_prev = a;
    }

    // Waypoints are reached exactly at the section boundaries
    for (size_t w = 0; w < input.intermediate_positions.size(); ++w) {
        trajectory.at_time(cumulative_times[w], p);
        for (size_t d = 0; d < dofs; ++d) {
            CHECK( p[d] == doctest::Approx(input.intermediate_positions[w][d]).epsilon(1e-7) );
        }
    }

    // Target state
    trajectory.at_time(duration, p, v, a);
    for (size_t d = 0; d < dofs; ++d) {
        CHECK( p[d] == doctest::Approx(input.target_position[d]).epsilon(1e-7) );
        CHECK( v[d] == doctest::Approx(input.target_velocity[d]).epsilon(1e-7) );
        CHECK( a[d] == doctest::Approx(input.target_acceleration[d]).epsilon(1e-7) );
    }
}


//! Duration of the trajectory that stops at every waypoint (with zero velocity and acceleration)
template<size_t DOFs>
double stop_at_waypoints_duration(const InputParameter<DOFs>& input) {
    const size_t dofs = input.degrees_of_freedom;
    auto make_otg = [&]() {
        if constexpr (DOFs == 0) {
            return Ruckig<DOFs>(dofs, 0.01);
        } else {
            return Ruckig<DOFs>(0.01);
        }
    };
    auto make_trajectory = [&]() {
        if constexpr (DOFs == 0) {
            return Trajectory<DOFs>(dofs);
        } else {
            return Trajectory<DOFs>();
        }
    };
    auto otg = make_otg();
    InputParameter<DOFs> section = input;
    section.intermediate_positions.clear();
    section.per_section_max_velocity = std::nullopt;
    section.per_section_max_acceleration = std::nullopt;
    section.per_section_max_jerk = std::nullopt;
    section.per_section_min_velocity = std::nullopt;
    section.per_section_min_acceleration = std::nullopt;
    section.per_section_minimum_duration = std::nullopt;

    std::vector<std::vector<double>> positions;
    positions.push_back(std::vector<double>(input.current_position.begin(), input.current_position.end()));
    for (const auto& wp: input.intermediate_positions) {
        positions.push_back(std::vector<double>(wp.begin(), wp.end()));
    }
    positions.push_back(std::vector<double>(input.target_position.begin(), input.target_position.end()));

    double duration {0.0};
    for (size_t k = 0; k + 1 < positions.size(); ++k) {
        for (size_t d = 0; d < dofs; ++d) {
            section.current_position[d] = positions[k][d];
            section.target_position[d] = positions[k + 1][d];
            section.current_velocity[d] = (k == 0) ? input.current_velocity[d] : 0.0;
            section.current_acceleration[d] = (k == 0) ? input.current_acceleration[d] : 0.0;
            section.target_velocity[d] = (k + 2 == positions.size()) ? input.target_velocity[d] : 0.0;
            section.target_acceleration[d] = (k + 2 == positions.size()) ? input.target_acceleration[d] : 0.0;
        }
        auto trajectory = make_trajectory();
        REQUIRE( otg.calculate(section, trajectory) == Result::Working );
        duration += trajectory.get_duration();
    }
    return duration;
}


InputParameter<DynamicDOFs> example_input() {
    InputParameter<DynamicDOFs> input(3);
    input.current_position = {0.2, 0.0, -0.3};
    input.current_velocity = {0.0, 0.2, 0.0};
    input.current_acceleration = {0.0, 0.6, 0.0};
    input.intermediate_positions = {{1.4, -1.6, 1.0}, {-0.6, -0.5, 0.4}, {-0.4, -0.35, 0.0}, {0.8, 1.8, -0.1}};
    input.target_position = {0.5, 1.0, 0.0};
    input.target_velocity = {0.2, 0.0, 0.3};
    input.target_acceleration = {0.0, 0.1, -0.1};
    input.max_velocity = {1.0, 2.0, 1.0};
    input.max_acceleration = {3.0, 2.0, 2.0};
    input.max_jerk = {6.0, 10.0, 20.0};
    return input;
}


TEST_CASE("waypoints-example") {
    const auto input = example_input();
    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    otg.set_waypoints_backend(WaypointsBackend::Local);

    Trajectory<DynamicDOFs> trajectory(3, 10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );
    check_waypoint_trajectory(input, trajectory);

    // Faster than stopping at every waypoint, and at least as fast as the cloud API (7.87 s)
    CHECK( trajectory.get_duration() < stop_at_waypoints_duration(input) );
    CHECK( trajectory.get_duration() < 7.9 );
}


TEST_CASE("waypoints-online") {
    auto input = example_input();
    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    OutputParameter<DynamicDOFs> output(3, 10);

    size_t steps {0}, section_changes {0}, calculations {0};
    Result result;
    while ((result = otg.update(input, output)) == Result::Working) {
        output.pass_to_input(input);
        steps += 1;
        section_changes += output.did_section_change ? 1 : 0;
        calculations += output.new_calculation ? 1 : 0;
        REQUIRE( steps < 10000 );
    }
    CHECK( result == Result::Finished );
    CHECK( calculations == 1 );
    CHECK( section_changes == 4 );
    CHECK( input.intermediate_positions.empty() );
    CHECK( output.time > output.trajectory.get_duration() );

    // After the end of the trajectory, the target state is kept (with constant acceleration)
    const double t_after = output.time - output.trajectory.get_duration();
    const auto [p0, v0, a0] = integrate(t_after, 0.5, 0.2, 0.0, 0.0);
    const auto [p1, v1, a1] = integrate(t_after, 1.0, 0.0, 0.1, 0.0);
    const auto [p2, v2, a2] = integrate(t_after, 0.0, 0.3, -0.1, 0.0);
    CHECK( output.new_position[0] == doctest::Approx(p0).epsilon(1e-6) );
    CHECK( output.new_position[1] == doctest::Approx(p1).epsilon(1e-6) );
    CHECK( output.new_position[2] == doctest::Approx(p2).epsilon(1e-6) );
    CHECK( output.new_velocity[0] == doctest::Approx(v0).epsilon(1e-6) );
    CHECK( output.new_velocity[1] == doctest::Approx(v1).epsilon(1e-6) );
    CHECK( output.new_velocity[2] == doctest::Approx(v2).epsilon(1e-6) );
}


TEST_CASE("waypoints-fixed-dofs") {
    Ruckig<3> otg(0.01, 10);
    InputParameter<3> input;
    input.current_position = {0.2, 0.0, -0.3};
    input.intermediate_positions = {{1.4, -1.6, 1.0}, {-0.6, -0.5, 0.4}};
    input.target_position = {0.5, 1.0, 0.0};
    input.max_velocity = {1.0, 2.0, 1.0};
    input.max_acceleration = {3.0, 2.0, 2.0};
    input.max_jerk = {6.0, 10.0, 20.0};

    Trajectory<3> trajectory(10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );
    check_waypoint_trajectory(input, trajectory);
    CHECK( trajectory.get_duration() < stop_at_waypoints_duration(input) );
}


TEST_CASE("waypoints-single-dof") {
    Ruckig<1> otg(0.01, 10);
    InputParameter<1> input;
    input.current_position = {0.0};
    input.intermediate_positions = {{1.0}, {2.0}, {1.5}, {3.0}};
    input.target_position = {2.0};
    input.max_velocity = {1.0};
    input.max_acceleration = {2.0};
    input.max_jerk = {10.0};

    Trajectory<1> trajectory(10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );
    check_waypoint_trajectory(input, trajectory);
    CHECK( trajectory.get_duration() < stop_at_waypoints_duration(input) );
}


TEST_CASE("waypoints-per-section-limits") {
    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    InputParameter<DynamicDOFs> input(3);
    input.current_position = {0.8, 0.0, 0.5};
    input.intermediate_positions = {{1.4, -1.6, 1.0}, {-0.6, -0.5, 0.4}, {-0.4, -0.35, 0.0}, {-0.2, 0.35, -0.1}, {0.2, 0.5, -0.1}, {0.8, 1.8, -0.1}};
    input.target_position = {0.5, 1.2, 0.0};
    input.max_velocity = {3.0, 2.0, 2.0};
    input.max_acceleration = {6.0, 4.0, 4.0};
    input.max_jerk = {16.0, 10.0, 20.0};

    SUBCASE("minimum duration") {
        input.per_section_minimum_duration = std::vector<double>{0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0};

        Trajectory<DynamicDOFs> trajectory(3, 10);
        REQUIRE( otg.calculate(input, trajectory) == Result::Working );
        check_waypoint_trajectory(input, trajectory);

        const auto times = trajectory.get_intermediate_durations();
        CHECK( times[1] - times[0] >= 2.0 - 1e-9 );
        CHECK( times[3] - times[2] >= 1.0 - 1e-9 );
        CHECK( times[5] - times[4] >= 2.0 - 1e-9 );
    }

    SUBCASE("velocity limits") {
        std::vector<std::vector<double>> max_velocity(7, {3.0, 2.0, 2.0});
        max_velocity[2] = {0.5, 0.5, 0.5};
        input.per_section_max_velocity = max_velocity;

        Trajectory<DynamicDOFs> trajectory(3, 10);
        REQUIRE( otg.calculate(input, trajectory) == Result::Working );
        check_waypoint_trajectory(input, trajectory);

        // Velocity in section 2 stays within its tighter limit
        const auto times = trajectory.get_intermediate_durations();
        std::vector<double> p(3), v(3), a(3);
        for (double t = times[1]; t <= times[2]; t += 1e-3) {
            trajectory.at_time(t, p, v, a);
            for (size_t d = 0; d < 3; ++d) {
                CHECK( std::abs(v[d]) <= 0.5 + 1e-6 );
            }
        }
    }
}


TEST_CASE("waypoints-positional-limits") {
    Ruckig<DynamicDOFs> otg(2, 0.01, 10);
    InputParameter<DynamicDOFs> input(2);
    input.current_position = {0.0, 0.0};
    input.intermediate_positions = {{1.0, 0.5}, {0.0, 1.0}};
    input.target_position = {1.0, 1.5};
    input.max_velocity = {1.0, 1.0};
    input.max_acceleration = {2.0, 2.0};
    input.max_jerk = {10.0, 10.0};

    Trajectory<DynamicDOFs> trajectory(2, 10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );

    input.max_position = {0.5, 10.0};
    CHECK( otg.calculate(input, trajectory) == Result::ErrorPositionalLimits );
}


TEST_CASE("waypoints-interrupt") {
    auto input = example_input();
    input.interrupt_calculation_duration = 50.0; // [µs]

    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    Trajectory<DynamicDOFs> trajectory(3, 10);
    bool was_interrupted {false};
    REQUIRE( otg.calculate(input, trajectory, was_interrupted) == Result::Working );
    check_waypoint_trajectory(input, trajectory);
    CHECK( trajectory.get_duration() <= stop_at_waypoints_duration(input) + 1e-9 );
}


TEST_CASE("waypoints-warm-start") {
    const auto input = example_input();
    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    Trajectory<DynamicDOFs> trajectory(3, 10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );
    const double duration = trajectory.get_duration();

    // Extract the waypoint states of the solution
    const auto profiles = trajectory.get_profiles();
    std::vector<std::vector<double>> velocities, accelerations;
    for (size_t s = 0; s + 1 < profiles.size(); ++s) {
        std::vector<double> v(3), a(3);
        for (size_t d = 0; d < 3; ++d) {
            v[d] = profiles[s][d].vf;
            a[d] = profiles[s][d].af;
        }
        velocities.push_back(v);
        accelerations.push_back(a);
    }

    SUBCASE("calculate from states") {
        Trajectory<DynamicDOFs> replay(3, 10);
        REQUIRE( otg.calculator.local_waypoints_calculator.calculate_from_states<false>(input, replay, 0.01, velocities, accelerations) == Result::Working );
        check_waypoint_trajectory(input, replay);
        CHECK( replay.get_duration() == doctest::Approx(duration).epsilon(1e-9) );

        // Stopping at every waypoint
        std::vector<std::vector<double>> zeros(velocities.size(), std::vector<double>(3, 0.0));
        REQUIRE( otg.calculator.local_waypoints_calculator.calculate_from_states<false>(input, replay, 0.01, zeros, zeros) == Result::Working );
        check_waypoint_trajectory(input, replay);
        CHECK( replay.get_duration() == doctest::Approx(stop_at_waypoints_duration(input)).epsilon(1e-9) );

        // Wrong number of states
        std::vector<std::vector<double>> wrong(2, std::vector<double>(3, 0.0));
        CHECK( otg.calculator.local_waypoints_calculator.calculate_from_states<false>(input, replay, 0.01, wrong, wrong) == Result::ErrorInvalidInput );
    }

    SUBCASE("warm start") {
        auto& calculator = otg.calculator.local_waypoints_calculator;
        calculator.number_global_steps = 0;
        calculator.number_local_steps = 4;
        calculator.initial_velocities = velocities;
        calculator.initial_accelerations = accelerations;

        Trajectory<DynamicDOFs> warm(3, 10);
        REQUIRE( otg.calculate(input, warm) == Result::Working );
        check_waypoint_trajectory(input, warm);
        CHECK( warm.get_duration() <= duration + 1e-9 );
    }
}


TEST_CASE("waypoints-disabled-dof") {
    auto input = example_input();
    input.enabled = {true, false, true};
    input.current_velocity[1] = 0.1;
    input.current_acceleration[1] = 0.0;

    Ruckig<DynamicDOFs> otg(3, 0.01, 10);
    Trajectory<DynamicDOFs> trajectory(3, 10);
    REQUIRE( otg.calculate(input, trajectory) == Result::Working );

    // The disabled DoF keeps its constant velocity, the others reach the waypoints
    const auto times = trajectory.get_intermediate_durations();
    std::vector<double> p(3), v(3), a(3);
    for (size_t w = 0; w < input.intermediate_positions.size(); ++w) {
        trajectory.at_time(times[w], p, v, a);
        CHECK( p[0] == doctest::Approx(input.intermediate_positions[w][0]).epsilon(1e-7) );
        CHECK( p[2] == doctest::Approx(input.intermediate_positions[w][2]).epsilon(1e-7) );
        CHECK( p[1] == doctest::Approx(0.1 * times[w]).epsilon(1e-6) );
        CHECK( v[1] == doctest::Approx(0.1).epsilon(1e-9) );
    }
}


TEST_CASE("waypoints-random") {
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dofs_dist(1, 5);
    std::uniform_int_distribution<size_t> waypoints_dist(1, 6);
    std::normal_distribution<double> position_dist(0.0, 1.0);
    std::uniform_real_distribution<double> vel_limit_dist(0.3, 3.0);
    std::uniform_real_distribution<double> acc_limit_dist(0.5, 8.0);
    std::uniform_real_distribution<double> jerk_limit_dist(2.0, 50.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (size_t i = 0; i < 64; ++i) {
        const size_t dofs = dofs_dist(rng);
        const size_t n_waypoints = waypoints_dist(rng);

        InputParameter<DynamicDOFs> input(dofs);
        for (size_t d = 0; d < dofs; ++d) {
            input.current_position[d] = position_dist(rng);
            input.target_position[d] = position_dist(rng);
            input.max_velocity[d] = vel_limit_dist(rng);
            input.max_acceleration[d] = acc_limit_dist(rng);
            input.max_jerk[d] = jerk_limit_dist(rng);
            if (unit(rng) < 0.3) {
                input.current_velocity[d] = 0.5 * input.max_velocity[d] * (2 * unit(rng) - 1);
                input.current_acceleration[d] = 0.5 * input.max_acceleration[d] * (2 * unit(rng) - 1);
            }
            if (unit(rng) < 0.3) {
                input.target_velocity[d] = 0.5 * input.max_velocity[d] * (2 * unit(rng) - 1);
                input.target_acceleration[d] = 0.5 * input.max_acceleration[d] * (2 * unit(rng) - 1);
            }
        }
        for (size_t w = 0; w < n_waypoints; ++w) {
            std::vector<double> wp(dofs);
            for (size_t d = 0; d < dofs; ++d) {
                wp[d] = position_dist(rng);
            }
            input.intermediate_positions.push_back(wp);
        }

        const std::string input_string = input.to_string();
        CAPTURE( i );
        CAPTURE( input_string );
        Ruckig<DynamicDOFs> otg(dofs, 0.01, n_waypoints);
        otg.calculator.local_waypoints_calculator.number_global_steps = 4;
        otg.calculator.local_waypoints_calculator.number_local_steps = 8;
        otg.calculator.local_waypoints_calculator.number_smoothing_steps = 1;

        Trajectory<DynamicDOFs> trajectory(dofs, n_waypoints);
        REQUIRE( otg.calculate(input, trajectory) == Result::Working );
        check_waypoint_trajectory(input, trajectory);
        CHECK( trajectory.get_duration() <= stop_at_waypoints_duration(input) + 1e-9 );
    }
}


int main(int argc, char** argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
