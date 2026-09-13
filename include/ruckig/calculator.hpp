#pragma once

#include <ruckig/calculator_target.hpp>
#ifdef WITH_CLOUD_CLIENT
#include <ruckig/calculator_cloud.hpp>
#endif
#ifdef WITH_LOCAL_WAYPOINTS
#include <ruckig/calculator_local.hpp>
#endif
#include <ruckig/error.hpp>
#include <ruckig/input_parameter.hpp>
#include <ruckig/trajectory.hpp>


namespace ruckig {

#ifdef RUCKIG_WITH_WAYPOINTS
//! Backend for calculating trajectories with intermediate waypoints
enum class WaypointsBackend {
    Local, ///< Local calculation on this machine (default if built with BUILD_LOCAL_WAYPOINTS)
    Cloud, ///< Remote calculation via the Ruckig cloud API (requires BUILD_CLOUD_CLIENT and network access)
};
#endif


//! Internal interface for the main calculator and its hyperparameters
template<size_t DOFs, template<class, size_t> class CustomVector = StandardVector>
class Calculator {
    inline bool use_waypoints_trajectory(const InputParameter<DOFs, CustomVector>& input) {
        return !input.intermediate_positions.empty() && input.control_interface == ControlInterface::Position;
    }

public:
    //! Calculator for state-to-state trajectories
    TargetCalculator<DOFs, CustomVector> target_calculator;

#if defined WITH_CLOUD_CLIENT
    //! Calculator for trajectories with intermediate waypoints via the cloud API
    WaypointsCalculator<DOFs, CustomVector> waypoints_calculator;
#endif

#if defined WITH_LOCAL_WAYPOINTS
    //! Local calculator for trajectories with intermediate waypoints (including its hyperparameters)
    LocalWaypointsCalculator<DOFs, CustomVector> local_waypoints_calculator;
#endif

#if defined RUCKIG_WITH_WAYPOINTS
    //! Backend used for trajectories with intermediate waypoints
    WaypointsBackend waypoints_backend {
#if defined WITH_LOCAL_WAYPOINTS
        WaypointsBackend::Local
#else
        WaypointsBackend::Cloud
#endif
    };
#endif

    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Calculator() { }

#if defined RUCKIG_WITH_WAYPOINTS
    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Calculator(size_t max_waypoints)
#if defined WITH_CLOUD_CLIENT
        : waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(max_waypoints))
#endif
    {
        (void)max_waypoints;
    }
#endif

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Calculator(size_t dofs):
        target_calculator(TargetCalculator<DOFs, CustomVector>(dofs))
#if defined WITH_CLOUD_CLIENT
        , waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(dofs))
#endif
#if defined WITH_LOCAL_WAYPOINTS
        , local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(dofs))
#endif
    { }

#if defined RUCKIG_WITH_WAYPOINTS
    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Calculator(size_t dofs, size_t max_waypoints):
        target_calculator(TargetCalculator<DOFs, CustomVector>(dofs))
#if defined WITH_CLOUD_CLIENT
        , waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(dofs, max_waypoints))
#endif
#if defined WITH_LOCAL_WAYPOINTS
        , local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(dofs, max_waypoints))
#endif
    {
        (void)max_waypoints;
    }
#endif

    //! Calculate the time-optimal waypoint-based trajectory
    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
#if defined RUCKIG_WITH_WAYPOINTS
        if (use_waypoints_trajectory(input)) {
            switch (waypoints_backend) {
#if defined WITH_LOCAL_WAYPOINTS
                case WaypointsBackend::Local: return local_waypoints_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
#if defined WITH_CLOUD_CLIENT
                case WaypointsBackend::Cloud: return waypoints_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
                default: {
                    if constexpr (throw_error) {
                        throw RuckigError("the selected waypoints backend is not available in this build.");
                    }
                    return Result::Error;
                }
            }
        }
#endif
        return target_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
    }

    //! Continue the trajectory calculation
    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
#if defined RUCKIG_WITH_WAYPOINTS
        if (use_waypoints_trajectory(input)) {
            switch (waypoints_backend) {
#if defined WITH_LOCAL_WAYPOINTS
                case WaypointsBackend::Local: return local_waypoints_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
#if defined WITH_CLOUD_CLIENT
                case WaypointsBackend::Cloud: return waypoints_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
                default: {
                    if constexpr (throw_error) {
                        throw RuckigError("the selected waypoints backend is not available in this build.");
                    }
                    return Result::Error;
                }
            }
        }
#endif
        return target_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
    }
};

} // namespace ruckig
