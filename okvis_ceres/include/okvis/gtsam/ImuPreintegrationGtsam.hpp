/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file ImuPreintegrationGtsam.hpp
 * @brief GTSAM combined IMU preintegration matching OKVIS's ImuError semantics.
 *
 * Maps okvis::ImuParameters onto gtsam::PreintegrationCombinedParams and
 * preintegrates an okvis::ImuMeasurementDeque into a
 * gtsam::PreintegratedCombinedMeasurements (15-DOF, models bias random walk),
 * ready to feed a gtsam::CombinedImuFactor.
 *
 * Conventions pinned (see TestGtsamImu.cpp):
 *  - Gravity: OKVIS world is z-up with g_W = +g*z used as -0.5*g_W*dt^2; the
 *    physical gravity is (0,0,-g). GTSAM MakeSharedU(g) sets n_gravity=(0,0,-g).
 *  - Noise: OKVIS adds dt*sigma_c^2 per step; GTSAM uses continuous-time PSDs
 *    (sigma_c^2) and divides by dt internally -> pass sigma_c^2 directly.
 *  - Integration: OKVIS uses the trapezoidal average of consecutive raw samples;
 *    we feed GTSAM that same average per sub-interval.
 */

#ifndef INCLUDE_OKVIS_GTSAM_IMUPREINTEGRATIONGTSAM_HPP_
#define INCLUDE_OKVIS_GTSAM_IMUPREINTEGRATIONGTSAM_HPP_

#include <boost/shared_ptr.hpp>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>

namespace okvis {
namespace gtsam_backend {

/// \brief Build GTSAM combined-preintegration parameters from OKVIS IMU params.
/// \param imuParameters OKVIS IMU noise/bias parameters.
/// \return shared params with z-up gravity and matched continuous-time PSDs.
boost::shared_ptr<gtsam::PreintegrationCombinedParams> makeCombinedParams(
    const okvis::ImuParameters& imuParameters);

/// \brief Preintegrate OKVIS IMU measurements over [t0, t1] for a given bias.
/// \param imuMeasurements IMU samples spanning at least [t0, t1].
/// \param imuParameters   OKVIS IMU parameters (for noise + saturation limits).
/// \param bias            Linearization bias for the preintegration.
/// \param t0              Start time.
/// \param t1              End time.
/// \return The preintegrated combined measurements.
gtsam::PreintegratedCombinedMeasurements preintegrate(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParameters,
    const gtsam::imuBias::ConstantBias& bias, const okvis::Time& t0,
    const okvis::Time& t1);

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_IMUPREINTEGRATIONGTSAM_HPP_
