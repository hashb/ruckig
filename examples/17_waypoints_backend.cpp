// This example shows how to switch between the local waypoints calculator and the cloud API
// for trajectories with intermediate waypoints. At least one of the backends has to be enabled
// at build time (BUILD_LOCAL_WAYPOINTS or BUILD_CLOUD_CLIENT).

#include <iostream>

#include <ruckig/ruckig.hpp>

#include "plotter.hpp"


using namespace ruckig;


void run(Ruckig<3>& otg, const InputParameter<3>& input, const char* label) {
    std::cout << "\n=== " << label << " ===" << std::endl;

    Trajectory<3> trajectory(10);
    const Result result = otg.calculate(input, trajectory);
    if (result < 0) {
        std::cout << "calculation failed with code " << static_cast<int>(result) << std::endl;
        return;
    }

    std::cout << "duration: " << trajectory.get_duration() << " [s]" << std::endl;
    std::cout << "waypoints reached at:";
    for (const double t: trajectory.get_intermediate_durations()) {
        std::cout << " " << t;
    }
    std::cout << " [s]" << std::endl;
}


int main() {
    const double control_cycle = 0.01;
    const size_t max_number_of_waypoints = 10;  // for memory allocation

    Ruckig<3> otg(control_cycle, max_number_of_waypoints);
    InputParameter<3> input;

    input.current_position = {0.2, 0.0, -0.3};
    input.current_velocity = {0.0, 0.2, 0.0};
    input.current_acceleration = {0.0, 0.6, 0.0};

    input.intermediate_positions = {
        {1.4, -1.6, 1.0},
        {-0.6, -0.5, 0.4},
        {-0.4, -0.35, 0.0},
        {0.8, 1.8, -0.1}
    };

    input.target_position = {0.5, 1.0, 0.0};
    input.target_velocity = {0.2, 0.0, 0.3};
    input.target_acceleration = {0.0, 0.1, -0.1};

    input.max_velocity = {1.0, 2.0, 1.0};
    input.max_acceleration = {3.0, 2.0, 2.0};
    input.max_jerk = {6.0, 10.0, 20.0};

#if defined WITH_LOCAL_WAYPOINTS
    // The hyperparameters of the local calculator trade calculation time for trajectory duration
    otg.calculator.local_waypoints_calculator.number_global_steps = 32;
    otg.calculator.local_waypoints_calculator.number_local_steps = 32;
    otg.calculator.local_waypoints_calculator.number_smoothing_steps = 4;

    otg.set_waypoints_backend(WaypointsBackend::Local);
    run(otg, input, "Local waypoints calculator");
#else
    std::cout << "[info] local waypoints calculator not built, skipping." << std::endl;
#endif

#if defined WITH_CLOUD_CLIENT
    otg.set_waypoints_backend(WaypointsBackend::Cloud);
    run(otg, input, "Cloud API");
#else
    std::cout << "[info] cloud client not built, skipping." << std::endl;
#endif

    return 0;
}
