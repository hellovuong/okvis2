/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGtsamDelayedMarg.cpp
 * @brief Validates GtsamBackend delayed marginalization (Phase D): rewriteAfterInit
 *        applies the gravity alignment, the re-marginalization swap keeps the
 *        window consistent, and the bias-change trigger honours throttle.
 */

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Rot3.h>

#include <okvis/GtsamBackend.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/kinematics/Transformation.hpp>

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

TEST(GtsamDelayedMarg, RewriteAfterInitRotatesStates) {
  const auto params = makeImuParameters();
  okvis::GtsamBackend backend(params);

  const okvis::kinematics::Transformation T_WS_visual(
      Eigen::Vector3d(1.0, 2.0, 3.0),
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())));
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = Eigen::Vector3d(0.5, -0.2, 0.1);
  backend.addState(okvis::StateId(1), T_WS_visual, sb);

  const gtsam::Rot3 R_gw = gtsam::Rot3::RzRyRx(0.0, 0.1, 0.2);
  const gtsam::imuBias::ConstantBias bias(Eigen::Vector3d(0.01, 0.02, 0.03),
                                          Eigen::Vector3d(0.001, 0.002, 0.003));
  const Eigen::Vector3d v_W(0.6, -0.1, 0.2);
  backend.rewriteAfterInit(R_gw, bias, {{1, v_W}});

  const okvis::kinematics::Transformation T = backend.getPose(okvis::StateId(1));
  const gtsam::Rot3 R_expected = R_gw * gtsam::Rot3(Eigen::Quaterniond(T_WS_visual.q()));
  const Eigen::Vector3d t_expected = R_gw.rotate(T_WS_visual.r());
  EXPECT_LT((T.r() - t_expected).norm(), 1e-9);
  EXPECT_LT(Eigen::Quaterniond(T.q()).angularDistance(R_expected.toQuaternion()), 1e-9);

  const okvis::SpeedAndBias sbOut = backend.getSpeedAndBias(okvis::StateId(1));
  EXPECT_LT((sbOut.head<3>() - R_gw.rotate(v_W)).norm(), 1e-9);  // velocity rotated
  EXPECT_LT((sbOut.segment<3>(3) - bias.gyroscope()).norm(), 1e-12);
  EXPECT_LT((sbOut.tail<3>() - bias.accelerometer()).norm(), 1e-12);
}

TEST(GtsamDelayedMarg, RemarginalizeKeepsWindowConsistent) {
  const auto params = makeImuParameters();
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());

  const int nFrames = 3;
  const double dt = 0.1;
  const Eigen::Vector3d v_W(1.0, 0.0, 0.0);
  std::vector<okvis::kinematics::Transformation> T_WS_gt(nFrames);
  for (int k = 0; k < nFrames; ++k)
    T_WS_gt[k] = okvis::kinematics::Transformation(v_W * (k * dt),
                                                   Eigen::Quaterniond::Identity());
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
    m.measurement.accelerometers = Eigen::Vector3d(0, 0, params.g);
    imu.push_back(m);
  }

  okvis::GtsamBackend backend(params);
  backend.enableDelayedMarginalization(10);  // large lag: retain raw factors
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

  backend.optimise(20);
  backend.marginalizeState(okvis::StateId(1));  // active prior; delayed retains raw
  backend.remarginalize();  // re-derive the boundary prior from retained factors

  const double err = backend.optimise(20);
  EXPECT_TRUE(std::isfinite(err));
  for (int k = 1; k < nFrames; ++k) {
    const okvis::kinematics::Transformation T = backend.getPose(okvis::StateId(k + 1));
    EXPECT_LT((T.r() - T_WS_gt[k].r()).norm(), 5e-3) << "frame " << k;
  }
}

TEST(GtsamDelayedMarg, BiasTriggerHonoursThrottle) {
  const auto params = makeImuParameters();
  okvis::GtsamBackend backend(params);
  backend.enableDelayedMarginalization(10);
  // Add a couple of states + a marginalization so a retained boundary exists.
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  backend.addState(okvis::StateId(1), okvis::kinematics::Transformation(), sb);
  backend.addState(okvis::StateId(2), okvis::kinematics::Transformation(), sb);
  backend.addPosePrior(okvis::StateId(1), okvis::kinematics::Transformation(), 1e-3, 1e-3);
  backend.addPosePrior(okvis::StateId(2), okvis::kinematics::Transformation(), 1e-3, 1e-3);

  const gtsam::imuBias::ConstantBias b0;  // zero
  const gtsam::imuBias::ConstantBias b1(Eigen::Vector3d(0.3, 0, 0),
                                        Eigen::Vector3d::Zero());

  // First call only sets the baseline.
  EXPECT_FALSE(backend.maybeRemarginalize(b0, 0.0, 0.1, 2.0));
  // No change -> no trigger.
  EXPECT_FALSE(backend.maybeRemarginalize(b0, 1.0, 0.1, 2.0));
  // Big change but within the throttle interval -> suppressed.
  EXPECT_FALSE(backend.maybeRemarginalize(b1, 1.5, 0.1, 2.0));
  // Big change, interval elapsed -> fires.
  EXPECT_TRUE(backend.maybeRemarginalize(b1, 10.0, 0.1, 2.0));
}
