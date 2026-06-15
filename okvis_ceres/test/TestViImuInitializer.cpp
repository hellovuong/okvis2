/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestViImuInitializer.cpp
 * @brief Validates the DM-VIO dynamic IMU initializer: state transitions, the
 *        static-strategy fallback, and recovery of gravity / velocities / bias
 *        from an excited, tilted trajectory.
 */

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Rot3.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/ViImuInitializer.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace {

okvis::ImuParameters makeImuParameters(okvis::ImuParameters::InitStrategy strategy) {
  okvis::ImuParameters p;
  p.use = true;
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
  p.initStrategy = strategy;
  p.excitationThreshAcc = 0.5;
  p.excitationThreshGyr = 0.05;
  p.jointInitWindow = 16;
  p.initMinCondition = 1e-4;
  p.initTimeoutSec = 5.0;
  return p;
}

// Build a time-varying (excited) IMU stream.
okvis::ImuMeasurementDeque makeExcitedImu(double g, double rate, double duration) {
  okvis::ImuMeasurementDeque d;
  const double dt = 1.0 / rate;
  const int n = static_cast<int>(duration * rate) + 2;
  for (int i = 0; i < n; ++i) {
    const double t = i * dt;
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(t);
    m.measurement.gyroscopes =
        Eigen::Vector3d(0.3 * std::sin(2.0 * t), -0.25 * std::cos(1.5 * t),
                        0.2 * std::sin(t));
    m.measurement.accelerometers =
        Eigen::Vector3d(1.0 * std::sin(t), 0.8 * std::cos(2.0 * t), g);
    d.push_back(m);
  }
  return d;
}

}  // namespace

TEST(ViImuInitializer, StaticStrategyConvergesImmediately) {
  const auto params = makeImuParameters(okvis::ImuParameters::InitStrategy::Static);
  okvis::ViImuInitializer init(params);
  okvis::ImuMeasurementDeque imu = makeExcitedImu(params.g, 200.0, 0.2);

  init.addKeyframe(okvis::StateId(1), okvis::kinematics::Transformation(), imu,
                   okvis::Time(0.0), okvis::Time(0.0));
  const auto r = init.step();
  EXPECT_TRUE(r.converged);
  EXPECT_TRUE(r.usedStaticFallback);
  EXPECT_EQ(init.state(), okvis::ViImuInitializer::State::Converged);
}

TEST(ViImuInitializer, DynamicRecoversGravityAndVelocities) {
  const auto params = makeImuParameters(okvis::ImuParameters::InitStrategy::Dynamic);
  okvis::ImuMeasurementDeque imu = makeExcitedImu(params.g, 200.0, 2.0);

  // Ground-truth trajectory in the gravity-aligned frame G via propagation.
  const double dt = 0.1;
  const int nFrames = 16;
  std::vector<okvis::kinematics::Transformation> T_G;
  std::vector<Eigen::Vector3d> v_G;
  std::vector<okvis::Time> times;
  okvis::kinematics::Transformation T;
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = Eigen::Vector3d(0.3, 0.1, 0.0);
  T_G.push_back(T);
  v_G.push_back(sb.head<3>());
  times.push_back(okvis::Time(0.0));
  for (int k = 1; k < nFrames; ++k) {
    okvis::ceres::ImuError::propagation(imu, params, T, sb, okvis::Time((k - 1) * dt),
                                        okvis::Time(k * dt));
    T_G.push_back(T);
    v_G.push_back(sb.head<3>());
    times.push_back(okvis::Time(k * dt));
  }

  // Tilt into the visual world frame W by R_align (= ground-truth R_wg).
  const gtsam::Rot3 R_align = gtsam::Rot3::RzRyRx(0.0, 0.12, 0.18);
  std::vector<okvis::kinematics::Transformation> T_W;
  std::vector<Eigen::Vector3d> v_W;
  for (int k = 0; k < nFrames; ++k) {
    const gtsam::Rot3 Rg(Eigen::Quaterniond(T_G[k].q()));
    const Eigen::Matrix3d Rw = (R_align * Rg).matrix();
    T_W.emplace_back(R_align.rotate(T_G[k].r()), Eigen::Quaterniond(Rw));
    v_W.push_back(R_align.rotate(v_G[k]));
  }

  okvis::ViImuInitializer init(params);
  okvis::ViImuInitializer::Result result;
  for (int k = 0; k < nFrames; ++k) {
    const okvis::Time tPrev = (k == 0) ? times[0] : times[k - 1];
    init.addKeyframe(okvis::StateId(k + 1), T_W[k], imu, tPrev, times[k]);
    const auto r = init.step();
    if (r.converged) {
      result = r;
      break;
    }
  }

  ASSERT_TRUE(result.converged);
  EXPECT_FALSE(result.usedStaticFallback);

  // Gravity direction recovered (yaw-invariant).
  const Eigen::Vector3d gW_est = result.R_wg.rotate(Eigen::Vector3d(0, 0, -params.g));
  const Eigen::Vector3d gW_true = R_align.rotate(Eigen::Vector3d(0, 0, -params.g));
  const double angle = std::acos(std::min(
      1.0, gW_est.normalized().dot(gW_true.normalized())));
  EXPECT_LT(angle, 0.05) << "gravity dir off by " << angle << " rad";

  // Bias near zero (true bias was zero).
  EXPECT_LT(result.bias.vector().norm(), 0.1) << "bias: " << result.bias.vector().transpose();

  // Per-keyframe velocity recovery (check a mid-window frame present in result).
  int checked = 0;
  for (int k = 0; k < nFrames; ++k) {
    auto it = result.velocities.find(static_cast<std::uint64_t>(k + 1));
    if (it != result.velocities.end()) {
      EXPECT_LT((it->second - v_W[k]).norm(), 0.15) << "velocity " << k << " off";
      ++checked;
    }
  }
  EXPECT_GT(checked, 2);
}
