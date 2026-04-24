#pragma once

#include <ruckig/calculator_target.hpp>
#ifdef WITH_CLOUD_CLIENT
#include <ruckig/calculator_cloud.hpp>
#endif
#ifdef WITH_LOCAL_WAYPOINTS
#include <ruckig/calculator_local.hpp>
#endif
#include <ruckig/input_parameter.hpp>
#include <ruckig/trajectory.hpp>


namespace ruckig {

#if defined(WITH_CLOUD_CLIENT) || defined(WITH_LOCAL_WAYPOINTS)
//! Backend used to compute trajectories with intermediate waypoints.
enum class WaypointsBackend {
    //! Closed-source Ruckig Pro implementation accessed through the cloud API.
    Cloud,
    //! Offline implementation based on Kiemel & Kröger 2024 (arXiv:2407.13423).
    Local,
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
    //! Cloud-based waypoints calculator (Ruckig Pro)
    WaypointsCalculator<DOFs, CustomVector> waypoints_calculator;
#endif

#if defined WITH_LOCAL_WAYPOINTS
    //! Local waypoints calculator (Kiemel & Kröger 2024)
    LocalWaypointsCalculator<DOFs, CustomVector> local_waypoints_calculator;
#endif

#if defined(WITH_CLOUD_CLIENT) || defined(WITH_LOCAL_WAYPOINTS)
    //! Active backend used when intermediate positions are provided.
    //!
    //! Defaults to Local when the local backend is built (no network round
    //! trip required), otherwise Cloud. Set this field directly to switch.
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

#if defined(WITH_CLOUD_CLIENT) || defined(WITH_LOCAL_WAYPOINTS)
    template<size_t D = DOFs, typename std::enable_if<(D >= 1), int>::type = 0>
    explicit Calculator(size_t max_waypoints):
#if defined WITH_CLOUD_CLIENT
        waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(max_waypoints))
#if defined WITH_LOCAL_WAYPOINTS
        ,
        local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(max_waypoints))
#endif
#elif defined WITH_LOCAL_WAYPOINTS
        local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(max_waypoints))
#endif
        { (void)max_waypoints; }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Calculator(size_t dofs):
        target_calculator(TargetCalculator<DOFs, CustomVector>(dofs))
#if defined WITH_CLOUD_CLIENT
        ,
        waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(dofs))
#endif
#if defined WITH_LOCAL_WAYPOINTS
        ,
        local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(dofs))
#endif
        { }

    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Calculator(size_t dofs, size_t max_waypoints):
        target_calculator(TargetCalculator<DOFs, CustomVector>(dofs))
#if defined WITH_CLOUD_CLIENT
        ,
        waypoints_calculator(WaypointsCalculator<DOFs, CustomVector>(dofs, max_waypoints))
#endif
#if defined WITH_LOCAL_WAYPOINTS
        ,
        local_waypoints_calculator(LocalWaypointsCalculator<DOFs, CustomVector>(dofs, max_waypoints))
#endif
        { (void)max_waypoints; }
#else
    template<size_t D = DOFs, typename std::enable_if<(D == 0), int>::type = 0>
    explicit Calculator(size_t dofs): target_calculator(TargetCalculator<DOFs, CustomVector>(dofs)) { }
#endif

    //! Calculate the time-optimal waypoint-based trajectory
    template<bool throw_error>
    Result calculate(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
        Result result;
#if defined(WITH_CLOUD_CLIENT) || defined(WITH_LOCAL_WAYPOINTS)
        if (use_waypoints_trajectory(input)) {
            switch (waypoints_backend) {
#if defined WITH_LOCAL_WAYPOINTS
                case WaypointsBackend::Local:
                    return local_waypoints_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
#if defined WITH_CLOUD_CLIENT
                case WaypointsBackend::Cloud:
                    return waypoints_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
                default:
                    if constexpr (throw_error) {
                        throw RuckigError("selected waypoints backend is not enabled at build time.");
                    }
                    return Result::Error;
            }
        }
        result = target_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#else
        result = target_calculator.template calculate<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif

        return result;
    }

    //! Continue the trajectory calculation
    template<bool throw_error>
    Result continue_calculation(const InputParameter<DOFs, CustomVector>& input, Trajectory<DOFs, CustomVector>& trajectory, double delta_time, bool& was_interrupted) {
        Result result;
#if defined(WITH_CLOUD_CLIENT) || defined(WITH_LOCAL_WAYPOINTS)
        if (use_waypoints_trajectory(input)) {
            switch (waypoints_backend) {
#if defined WITH_LOCAL_WAYPOINTS
                case WaypointsBackend::Local:
                    return local_waypoints_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
#if defined WITH_CLOUD_CLIENT
                case WaypointsBackend::Cloud:
                    return waypoints_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif
                default:
                    if constexpr (throw_error) {
                        throw RuckigError("selected waypoints backend is not enabled at build time.");
                    }
                    return Result::Error;
            }
        }
        result = target_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#else
        result = target_calculator.template continue_calculation<throw_error>(input, trajectory, delta_time, was_interrupted);
#endif

        return result;
    }
};

} // namespace ruckig
