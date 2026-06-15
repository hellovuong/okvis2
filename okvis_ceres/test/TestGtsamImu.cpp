/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGtsamImu.cpp
 * @brief Validates GTSAM combined preintegration against okvis ImuError, and
 *        pins the gravity sign + bias ordering conventions.
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/navigation/NavState.h>

#include <okvis/Time.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/ImuError.hpp>

#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>

namespace {

okvis::ImuParameters makeImuParameters() {
  okvis::ImuParameters p;
  p.use = true;
  p.T_BS = okvis::kinematics::Transformation();  // identity
  p.a_max = 1000.0;
  p.g_max = 1000.0;
  p.sigma_g_c = 12.0e-4;
  p.sigma_a_c = 8.0e-3;
  p.sigma_gw_c = 4.0e-6;
  p.sigma_aw_c = 4.0e-5;
  p.sigma_bg = 0.03;
  p.sigma_ba = 0.1;
  p.g = 9.81;
  p.g0 = Eigen::Vector3d::Zero();
  p.a0 = Eigen::Vector3d::Zero();
  return p;
}

// Build a synthetic IMU stream with constant body-frame measurements.
okvis::ImuMeasurementDeque makeImuStream(const Eigen::Vector3d& gyro,
                                         const Eigen::Vector3d& acc,
                                         double rate_hz, double duration_s) {
  okvis::ImuMeasurementDeque deque;
  const double dt = 1.0 / rate_hz;
  const int n = static_cast<int>(duration_s * rate_hz) + 2;
  for (int i = 0; i < n; ++i) {
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(i * dt);
    m.measurement.gyroscopes = gyro;
    m.measurement.accelerometers = acc;
    deque.push_back(m);
  }
  return deque;
}

}  // namespace

// The bias conversion must survive a round-trip, with the [gyro,accel] (OKVIS)
// vs (accel,gyro) (GTSAM) ordering handled correctly.
TEST(GtsamImu, BiasOrderingRoundTrip) {
  okvis::SpeedAndBias sb;
  sb.head<3>() = Eigen::Vector3d(1.0, 2.0, 3.0);     // velocity
  sb.segment<3>(3) = Eigen::Vector3d(0.01, 0.02, 0.03);  // gyro bias
  sb.tail<3>() = Eigen::Vector3d(0.1, 0.2, 0.3);     // accel bias

  const auto bias = okvis::gtsam_backend::biasOf(sb);
  EXPECT_TRUE(bias.gyroscope().isApprox(sb.segment<3>(3)));
  EXPECT_TRUE(bias.accelerometer().isApprox(sb.tail<3>()));

  const okvis::SpeedAndBias sb2 =
      okvis::gtsam_backend::toSpeedAndBias(sb.head<3>(), bias);
  EXPECT_TRUE(sb2.isApprox(sb));
}

TEST(GtsamImu, PoseRoundTrip) {
  Eigen::Quaterniond q(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized()));
  okvis::kinematics::Transformation T(Eigen::Vector3d(1.0, -2.0, 3.0), q);
  const auto T2 = okvis::gtsam_backend::fromPose3(okvis::gtsam_backend::toPose3(T));
  EXPECT_TRUE(T2.r().isApprox(T.r(), 1e-12));
  EXPECT_NEAR(std::abs(T2.q().dot(T.q())), 1.0, 1e-12);
}

// Core validation: GTSAM preintegration + predict must match okvis propagation.
TEST(GtsamImu, MatchesOkvisPropagation) {
  const okvis::ImuParameters imuParameters = makeImuParameters();

  // True angular rate and specific force (body frame), plus a known bias.
  const Eigen::Vector3d omega_true(0.1, -0.2, 0.15);
  const Eigen::Vector3d b_g(0.01, -0.005, 0.008);
  const Eigen::Vector3d b_a(0.05, 0.02, -0.03);
  // Specific force: mostly gravity (z up reads +g) + small horizontal accel.
  const Eigen::Vector3d f_true(0.3, -0.1, 9.81);

  const Eigen::Vector3d gyro_meas = omega_true + b_g;
  const Eigen::Vector3d acc_meas = f_true + b_a;

  okvis::ImuMeasurementDeque imu = makeImuStream(gyro_meas, acc_meas, 200.0, 0.7);

  const okvis::Time t0(0.05);
  const okvis::Time t1(0.55);

  // okvis reference propagation.
  okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(0.0, 0.0, 0.0),
      Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())));
  okvis::SpeedAndBias sb0;
  sb0.head<3>() = Eigen::Vector3d(0.5, -0.3, 0.1);  // initial velocity
  sb0.segment<3>(3) = b_g;
  sb0.tail<3>() = b_a;

  okvis::kinematics::Transformation T_WS_1 = T_WS_0;
  okvis::SpeedAndBias sb1 = sb0;
  okvis::ceres::ImuError::propagation(imu, imuParameters, T_WS_1, sb1, t0, t1);

  // GTSAM preintegration + predict.
  const auto bias = okvis::gtsam_backend::biasOf(sb0);
  gtsam::PreintegratedCombinedMeasurements pim =
      okvis::gtsam_backend::preintegrate(imu, imuParameters, bias, t0, t1);
  const gtsam::NavState state0(okvis::gtsam_backend::toPose3(T_WS_0),
                               sb0.head<3>());
  const gtsam::NavState state1 = pim.predict(state0, bias);

  // Compare positions, rotations and velocities.
  const Eigen::Vector3d p_okvis = T_WS_1.r();
  const Eigen::Vector3d p_gtsam = state1.pose().translation();
  EXPECT_LT((p_okvis - p_gtsam).norm(), 1e-3)
      << "okvis p: " << p_okvis.transpose() << "\n"
      << "gtsam p: " << p_gtsam.transpose();

  const Eigen::Vector3d v_okvis = sb1.head<3>();
  const Eigen::Vector3d v_gtsam = state1.velocity();
  EXPECT_LT((v_okvis - v_gtsam).norm(), 5e-3)
      << "okvis v: " << v_okvis.transpose() << "\n"
      << "gtsam v: " << v_gtsam.transpose();

  const Eigen::Quaterniond q_okvis(T_WS_1.q());
  const Eigen::Quaterniond q_gtsam = state1.pose().rotation().toQuaternion();
  const double angle = q_okvis.angularDistance(q_gtsam);
  EXPECT_LT(angle, 2e-3) << "rotation mismatch: " << angle << " rad";
}
