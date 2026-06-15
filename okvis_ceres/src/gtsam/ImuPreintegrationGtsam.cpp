/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file ImuPreintegrationGtsam.cpp
 * @brief Implementation of GTSAM combined IMU preintegration for OKVIS.
 */

#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>

namespace okvis {
namespace gtsam_backend {

boost::shared_ptr<gtsam::PreintegrationCombinedParams> makeCombinedParams(
    const okvis::ImuParameters& imuParameters) {
  // z-up world, gravity = (0, 0, -g): matches OKVIS's g_W = +g*z used as
  // -0.5*g_W*dt^2 in the position residual.
  auto params =
      gtsam::PreintegrationCombinedParams::MakeSharedU(imuParameters.g);

  // OKVIS adds dt * sigma_c^2 of discrete noise per step; GTSAM stores the
  // continuous-time PSD (sigma_c^2) and divides by dt internally. So the
  // continuous covariances are simply the squared noise densities.
  const double g2 = imuParameters.sigma_g_c * imuParameters.sigma_g_c;
  const double a2 = imuParameters.sigma_a_c * imuParameters.sigma_a_c;
  params->setGyroscopeCovariance(g2 * Eigen::Matrix3d::Identity());
  params->setAccelerometerCovariance(a2 * Eigen::Matrix3d::Identity());

  // OKVIS has no explicit integration-uncertainty term; use a tiny value to
  // keep the information matrix well-conditioned.
  params->setIntegrationCovariance(1e-8 * Eigen::Matrix3d::Identity());

  // Bias random walk (continuous-time PSDs), matching OKVIS sigma_gw_c/sigma_aw_c.
  const double gw2 = imuParameters.sigma_gw_c * imuParameters.sigma_gw_c;
  const double aw2 = imuParameters.sigma_aw_c * imuParameters.sigma_aw_c;
  params->setBiasOmegaCovariance(gw2 * Eigen::Matrix3d::Identity());
  params->setBiasAccCovariance(aw2 * Eigen::Matrix3d::Identity());

  // Initial covariance of the integrated bias (cross term). OKVIS folds this
  // into the prior; keep small here.
  params->setBiasAccOmegaInt(1e-5 * gtsam::Matrix6::Identity());

  return params;
}

gtsam::PreintegratedCombinedMeasurements preintegrate(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParameters,
    const gtsam::imuBias::ConstantBias& bias, const okvis::Time& t0,
    const okvis::Time& t1) {
  gtsam::PreintegratedCombinedMeasurements pim(makeCombinedParams(imuParameters),
                                               bias);

  // Mirror okvis::ceres::ImuError::propagation: walk consecutive IMU samples,
  // clamp the first/last sub-interval to [t0, t1] with linear interpolation of
  // the measurements at the boundaries, and feed GTSAM the trapezoidal average
  // of the (interpolated) endpoint samples over each sub-interval. GTSAM
  // subtracts the bias internally, so we pass raw (bias-free) averages.
  const okvis::Time& start = t0;
  const okvis::Time& end = t1;
  okvis::Time time = start;
  bool hasStarted = false;

  for (auto it = imuMeasurements.begin(); it != imuMeasurements.end(); ++it) {
    auto next = it + 1;
    if (next == imuMeasurements.end()) break;  // need a pair to integrate

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = next->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = next->measurement.accelerometers;

    okvis::Time nexttime = next->timeStamp;
    double dt = (nexttime - time).toSec();

    // Clamp the tail of the interval to the end time, interpolating the sample.
    if (end < nexttime) {
      const double interval = (nexttime - it->timeStamp).toSec();
      nexttime = end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }

    // Clamp the head of the first integrated interval to the start time.
    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    const Eigen::Vector3d acc_avg = 0.5 * (acc_S_0 + acc_S_1);
    const Eigen::Vector3d gyro_avg = 0.5 * (omega_S_0 + omega_S_1);
    pim.integrateMeasurement(acc_avg, gyro_avg, dt);

    time = nexttime;
    if (nexttime == end) break;
  }

  return pim;
}

}  // namespace gtsam_backend
}  // namespace okvis
