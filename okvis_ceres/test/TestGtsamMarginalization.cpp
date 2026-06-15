/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGtsamMarginalization.cpp
 * @brief Validates GTSAM Schur-complement marginalization: (1) marginalizing a
 *        variable preserves the marginal covariance of the remaining variables
 *        on a pose chain; (2) marginalizing the oldest keyframe of a stereo+IMU
 *        window leaves the remaining states at ground truth after re-optimising.
 */

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <okvis/GtsamBackend.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/gtsam/Marginalization.hpp>
#include <okvis/kinematics/Transformation.hpp>

using gtsam::symbol_shorthand::X;

TEST(GtsamMarginalization, PoseChainPreservesMarginalCovariance) {
  const gtsam::Pose3 p1(gtsam::Rot3(), gtsam::Point3(0, 0, 0));
  const gtsam::Pose3 p2(gtsam::Rot3::Rz(0.1), gtsam::Point3(1, 0, 0));
  const gtsam::Pose3 p3(gtsam::Rot3::Rz(0.2), gtsam::Point3(2, 0.1, 0));

  gtsam::Values values;
  values.insert(X(1), p1);
  values.insert(X(2), p2);
  values.insert(X(3), p3);

  const auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(6, 0.01);
  const auto betweenNoise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);

  gtsam::NonlinearFactorGraph full;
  full.addPrior(X(1), p1, priorNoise);                                  // index 0
  full.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(              // index 1
      X(1), X(2), p1.between(p2), betweenNoise);
  full.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(              // index 2
      X(2), X(3), p2.between(p3), betweenNoise);

  const gtsam::Marginals margFull(full, values);
  const Eigen::MatrixXd cov1Full = margFull.marginalCovariance(X(1));
  const Eigen::MatrixXd cov3Full = margFull.marginalCovariance(X(3));

  // Marginalize out X(2): the two between factors are connected to it.
  gtsam::NonlinearFactorGraph touching;
  touching.push_back(full.at(1));
  touching.push_back(full.at(2));
  const gtsam::NonlinearFactor::shared_ptr prior =
      okvis::gtsam_backend::marginalizeOut(touching, values, {X(2)});
  ASSERT_TRUE(prior != nullptr);

  gtsam::NonlinearFactorGraph marg;
  marg.push_back(full.at(0));  // keep the prior on X(1)
  marg.push_back(prior);

  gtsam::Values valuesMarg;
  valuesMarg.insert(X(1), p1);
  valuesMarg.insert(X(3), p3);

  const gtsam::Marginals margMarg(marg, valuesMarg);
  const Eigen::MatrixXd cov1Marg = margMarg.marginalCovariance(X(1));
  const Eigen::MatrixXd cov3Marg = margMarg.marginalCovariance(X(3));

  EXPECT_LT((cov1Full - cov1Marg).norm(), 1e-6 * cov1Full.norm());
  EXPECT_LT((cov3Full - cov3Marg).norm(), 1e-6 * cov3Full.norm());
}

namespace {

typedef okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>
    CameraGeometry;

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

}  // namespace

TEST(GtsamMarginalization, BackendMarginalizeOldestKeyframe) {
  const okvis::ImuParameters imuParameters = makeImuParameters();
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());

  const int nFrames = 3;
  const double dt = 0.1;
  const Eigen::Vector3d v_W(1.0, 0.0, 0.0);
  std::vector<okvis::kinematics::Transformation> T_WS_gt(nFrames);
  for (int k = 0; k < nFrames; ++k) {
    T_WS_gt[k] = okvis::kinematics::Transformation(v_W * (k * dt),
                                                   Eigen::Quaterniond::Identity());
  }
  okvis::SpeedAndBias sb_gt;
  sb_gt.head<3>() = v_W;
  sb_gt.tail<6>().setZero();

  std::vector<okvis::kinematics::Transformation> T_SC(2);
  T_SC[0] = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Quaterniond::Identity());
  T_SC[1] = okvis::kinematics::Transformation(Eigen::Vector3d(0.11, 0, 0),
                                              Eigen::Quaterniond::Identity());

  std::vector<Eigen::Vector4d> lm_gt;
  for (int ix = -2; ix <= 2; ++ix)
    for (int iz = 0; iz < 2; ++iz) {
      Eigen::Vector4d p;
      p << 0.4 * ix, 0.2, 3.0 + iz, 1.0;
      lm_gt.push_back(p);
    }

  okvis::ImuMeasurementDeque imu;
  for (int i = 0; i < static_cast<int>((nFrames - 1) * dt / 0.005) + 4; ++i) {
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(i * 0.005);
    m.measurement.gyroscopes = Eigen::Vector3d::Zero();
    m.measurement.accelerometers = Eigen::Vector3d(0, 0, imuParameters.g);
    imu.push_back(m);
  }

  okvis::GtsamBackend backend(imuParameters);
  for (std::size_t c = 0; c < T_SC.size(); ++c) backend.setExtrinsics(c, T_SC[c], true);
  for (int k = 0; k < nFrames; ++k) backend.addState(okvis::StateId(k + 1), T_WS_gt[k], sb_gt);
  backend.addPosePrior(okvis::StateId(1), T_WS_gt[0], 1e-4, 1e-4);
  backend.addSpeedAndBiasPrior(okvis::StateId(1), sb_gt, 0.2, 0.01, 0.05);
  for (int k = 0; k < nFrames - 1; ++k)
    backend.addImuFactor(okvis::StateId(k + 1), okvis::StateId(k + 2), imu,
                         okvis::Time(k * dt), okvis::Time((k + 1) * dt));

  const Eigen::Matrix2d information =
      (Eigen::Matrix2d() << 64.0, 0.0, 0.0, 64.0).finished();
  for (int l = 0; l < static_cast<int>(lm_gt.size()); ++l) {
    backend.addLandmark(okvis::LandmarkId(1000 + l), lm_gt[l]);
    for (int k = 0; k < nFrames; ++k)
      for (std::size_t c = 0; c < T_SC.size(); ++c) {
        const Eigen::Vector4d hp_C =
            (T_SC[c].inverse().T() * T_WS_gt[k].inverse().T() * lm_gt[l]);
        Eigen::Vector2d kp;
        if (camera->projectHomogeneous(hp_C, &kp) ==
            okvis::cameras::ProjectionStatus::Successful)
          backend.addObservation<CameraGeometry>(okvis::StateId(k + 1),
                                                  okvis::LandmarkId(1000 + l), c, kp,
                                                  information, camera);
      }
  }

  // Optimise the full window, then marginalize the oldest keyframe.
  backend.optimise(20);
  ASSERT_TRUE(backend.hasState(okvis::StateId(1)));
  backend.marginalizeState(okvis::StateId(1));
  EXPECT_FALSE(backend.hasState(okvis::StateId(1)));

  // The remaining states should stay at ground truth after re-optimising.
  const double err = backend.optimise(20);
  EXPECT_TRUE(std::isfinite(err));
  for (int k = 1; k < nFrames; ++k) {
    const okvis::kinematics::Transformation T = backend.getPose(okvis::StateId(k + 1));
    EXPECT_LT((T.r() - T_WS_gt[k].r()).norm(), 5e-3) << "frame " << k;
    EXPECT_LT(Eigen::Quaterniond(T.q()).angularDistance(
                  Eigen::Quaterniond(T_WS_gt[k].q())), 5e-3) << "frame " << k;
  }
}
