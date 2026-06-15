/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGravityAlignmentFactor.cpp
 * @brief Validates GravityAlignmentFactor: near-zero residual at ground truth,
 *        and recovery of the gravity direction + velocities by optimisation.
 *
 * A consistent trajectory is generated with okvis ImuError::propagation in a
 * gravity-aligned frame G, then rotated into an arbitrary "visual" world frame W
 * by R_align (= the ground-truth R_wg). The factor must then recover the gravity
 * direction g_W = R_wg * (0,0,-g).
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/gtsam/GravityAlignmentFactor.hpp>
#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;  // reused here for the gravity key

namespace {

okvis::ImuParameters makeImuParameters() {
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
  return p;
}

okvis::ImuMeasurementDeque makeImuStream(const Eigen::Vector3d& gyro,
                                         const Eigen::Vector3d& acc, double rate,
                                         double duration) {
  okvis::ImuMeasurementDeque d;
  const double dt = 1.0 / rate;
  const int n = static_cast<int>(duration * rate) + 2;
  for (int i = 0; i < n; ++i) {
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(i * dt);
    m.measurement.gyroscopes = gyro;
    m.measurement.accelerometers = acc;
    d.push_back(m);
  }
  return d;
}

}  // namespace

class GravityAlignment : public ::testing::Test {
 protected:
  void SetUp() override {
    imuParameters_ = makeImuParameters();
    // Excited IMU stream (rotation + horizontal accel + gravity on z).
    imu_ = makeImuStream(Eigen::Vector3d(0.2, -0.1, 0.15),
                         Eigen::Vector3d(0.5, -0.3, imuParameters_.g), 200.0, 0.3);

    times_ = {okvis::Time(0.0), okvis::Time(0.1), okvis::Time(0.2)};

    // Ground-truth trajectory in the gravity-aligned frame G via propagation.
    okvis::kinematics::Transformation T = okvis::kinematics::Transformation();
    okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
    sb.head<3>() = Eigen::Vector3d(0.3, 0.0, 0.0);
    T_G_.push_back(T);
    v_G_.push_back(sb.head<3>());
    for (std::size_t k = 0; k + 1 < times_.size(); ++k) {
      okvis::ceres::ImuError::propagation(imu_, imuParameters_, T, sb, times_[k],
                                          times_[k + 1]);
      T_G_.push_back(T);
      v_G_.push_back(sb.head<3>());
    }

    // Rotate the whole problem into the "visual" world frame W by R_align.
    R_align_ = gtsam::Rot3::RzRyRx(0.0, 0.15, 0.2);  // yaw 0, pitch/roll tilt
    for (std::size_t k = 0; k < T_G_.size(); ++k) {
      const gtsam::Rot3 Rg(Eigen::Quaterniond(T_G_[k].q()));
      const gtsam::Pose3 Tg(Rg, T_G_[k].r());
      T_W_.push_back(gtsam::Pose3(R_align_ * Tg.rotation(),
                                  R_align_.rotate(Tg.translation())));
      v_W_.push_back(R_align_.rotate(v_G_[k]));
    }
  }

  okvis::ImuParameters imuParameters_;
  okvis::ImuMeasurementDeque imu_;
  std::vector<okvis::Time> times_;
  std::vector<okvis::kinematics::Transformation> T_G_;
  std::vector<Eigen::Vector3d> v_G_;
  std::vector<gtsam::Pose3> T_W_;
  std::vector<Eigen::Vector3d> v_W_;
  gtsam::Rot3 R_align_;
};

TEST_F(GravityAlignment, ResidualNearZeroAtTruth) {
  const gtsam::imuBias::ConstantBias bias;  // zero
  const auto pim01 = okvis::gtsam_backend::preintegrate(imu_, imuParameters_, bias,
                                                        times_[0], times_[1]);
  okvis::gtsam_backend::GravityAlignmentFactor f(
      okvis::gtsam_backend::GravityAlignmentFactor::makeNoiseModel(pim01), X(0),
      V(0), V(1), B(0), T_W_[0], T_W_[1], pim01, imuParameters_.g);

  const gtsam::Vector e = f.evaluateError(R_align_, v_W_[0], v_W_[1], bias);
  EXPECT_LT(e.norm(), 1e-2) << "residual: " << e.transpose();
}

TEST_F(GravityAlignment, RecoversGravityDirection) {
  const gtsam::imuBias::ConstantBias biasTrue;  // zero
  const auto pim01 = okvis::gtsam_backend::preintegrate(imu_, imuParameters_,
                                                        biasTrue, times_[0], times_[1]);
  const auto pim12 = okvis::gtsam_backend::preintegrate(imu_, imuParameters_,
                                                        biasTrue, times_[1], times_[2]);

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<okvis::gtsam_backend::GravityAlignmentFactor>(
      okvis::gtsam_backend::GravityAlignmentFactor::makeNoiseModel(pim01), X(0),
      V(0), V(1), B(0), T_W_[0], T_W_[1], pim01, imuParameters_.g);
  graph.emplace_shared<okvis::gtsam_backend::GravityAlignmentFactor>(
      okvis::gtsam_backend::GravityAlignmentFactor::makeNoiseModel(pim12), X(0),
      V(1), V(2), B(0), T_W_[1], T_W_[2], pim12, imuParameters_.g);

  // Pin the bias (unobservable under this excitation) and weakly regularize the
  // gravity gauge (yaw about gravity is unobservable).
  graph.addPrior(B(0), biasTrue,
                 gtsam::noiseModel::Isotropic::Sigma(6, 1e-4));
  graph.addPrior(X(0), R_align_, gtsam::noiseModel::Isotropic::Sigma(3, 1.0));

  // Perturbed initial guess.
  gtsam::Values values;
  values.insert(X(0), R_align_ * gtsam::Rot3::RzRyRx(0.1, -0.1, 0.08));
  values.insert(V(0), Eigen::Vector3d(v_W_[0] + Eigen::Vector3d(0.2, -0.1, 0.15)));
  values.insert(V(1), Eigen::Vector3d(v_W_[1] + Eigen::Vector3d(-0.1, 0.2, 0.05)));
  values.insert(V(2), Eigen::Vector3d(v_W_[2] + Eigen::Vector3d(0.05, 0.05, -0.1)));
  values.insert(B(0), biasTrue);

  gtsam::LevenbergMarquardtParams params;
  params.setMaxIterations(50);
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
  const gtsam::Values result = optimizer.optimize();

  // Recovered gravity direction must match the truth (yaw-invariant).
  const gtsam::Rot3 R_wg = result.at<gtsam::Rot3>(X(0));
  const Eigen::Vector3d gW_est =
      R_wg.rotate(Eigen::Vector3d(0, 0, -imuParameters_.g));
  const Eigen::Vector3d gW_true =
      R_align_.rotate(Eigen::Vector3d(0, 0, -imuParameters_.g));
  const double angle = std::acos(std::min(
      1.0, gW_est.normalized().dot(gW_true.normalized())));
  EXPECT_LT(angle, 0.03) << "gravity dir off by " << angle << " rad";

  // Velocities recovered.
  for (int k = 0; k < 3; ++k) {
    EXPECT_LT((result.at<Eigen::Vector3d>(V(k)) - v_W_[k]).norm(), 5e-2)
        << "velocity " << k << " off";
  }
}
