// This example shows how to switch between the cloud-based Ruckig Pro
// waypoints backend and the offline local backend implemented from
// Kiemel & Kroeger (arXiv:2407.13423, 2024). At least one of the two
// backends has to be enabled at build time (BUILD_CLOUD_CLIENT or
// BUILD_LOCAL_WAYPOINTS).

#include <iostream>

#include <ruckig/ruckig.hpp>

#include "plotter.hpp"


using namespace ruckig;


static void run(Ruckig<3>& otg, InputParameter<3>& input, const char* label) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    OutputParameter<3> output(10);

    Trajectory<3> trajectory(10);
    const Result calc_result = otg.calculate(input, trajectory);
    if (calc_result < 0) {
        std::cout << "[!] calculate() failed with code " << static_cast<int>(calc_result) << std::endl;
        return;
    }

    std::cout << "duration:           " << trajectory.get_duration() << " s" << std::endl;
    const auto sec = trajectory.get_intermediate_durations();
    std::cout << "section end times: ";
    for (auto t: sec) {
        std::cout << " " << t;
    }
    std::cout << std::endl;

    // Sample a couple of points from the trajectory.
    const double duration = trajectory.get_duration();
    for (int i = 0; i < 5; ++i) {
        const double t = duration * i / 4.0;
        std::array<double, 3> p, v, a;
        trajectory.at_time(t, p, v, a);
        std::cout << "  t=" << t << "  p=[" << pretty_print(p) << "]" << std::endl;
    }
}


int main() {
    const double control_cycle = 0.01;
    const size_t max_number_of_waypoints = 10;

    Ruckig<3> otg(control_cycle, max_number_of_waypoints);
    InputParameter<3> input;

    input.current_position = {0.0, 0.0, 0.0};
    input.current_velocity = {0.0, 0.0, 0.0};
    input.current_acceleration = {0.0, 0.0, 0.0};

    input.intermediate_positions = {
        { 1.0,  0.5, -0.2},
        { 0.4,  1.5,  0.6},
        {-0.3,  0.2,  1.0},
    };

    input.target_position = { 0.5, 1.0, 0.0};
    input.target_velocity = { 0.0, 0.0, 0.0};
    input.target_acceleration = {0.0, 0.0, 0.0};

    input.max_velocity = {1.0, 1.0, 1.0};
    input.max_acceleration = {2.0, 2.0, 2.0};
    input.max_jerk = {6.0, 6.0, 6.0};

#if defined WITH_LOCAL_WAYPOINTS
    otg.set_waypoints_backend(WaypointsBackend::Local);
    run(otg, input, "Local backend (Kiemel & Kroeger 2024)");
#else
    std::cout << "[info] local backend not built; skipping" << std::endl;
#endif

#if defined WITH_CLOUD_CLIENT
    otg.set_waypoints_backend(WaypointsBackend::Cloud);
    run(otg, input, "Cloud backend (Ruckig Pro)");
#else
    std::cout << "[info] cloud backend not built; skipping" << std::endl;
#endif

    return 0;
}
