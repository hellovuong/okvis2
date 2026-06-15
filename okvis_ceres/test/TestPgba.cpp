/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestPgba.cpp
 * @brief Proves DM-VIO Pose-Graph Bundle Adjustment initialization: landmarks
 *        are marginalized out of the stereo reprojection factors to form a
 *        visual pose-graph prior (capturing visual uncertainty), and PGBA then
 *        optimizes FREE poses + velocities + gravity (R_wg) + bias jointly
 *        against that prior plus the gravity-variable IMU factors. This is the
 *        ingredient that distinguishes DM-VIO's PGBA from inertial-only init.
 */

#include <gtest/gtest.h>

#include <vector>

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
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/GtsamReprojectionFactor.hpp>
#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>
#include <okvis/gtsam/Marginalization.hpp>
#include <okvis/gtsam/PgbaImuFactor.hpp>
#include <okvis/ViImuInitializer.hpp>

namespace gb = okvis::gtsam_backend;
using gtsam::symbol_shorthand::B;

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

// Build a visual pose-graph prior over poseKey(stateIds[k]) by adding stereo
// reprojection factors at the given (tilted, metric) poses and marginalizing out
// the landmarks + extrinsics. Mirrors what the live backend would hand the init.
gtsam::NonlinearFactor::shared_ptr buildVisualPosePrior(
    const std::vector<std::uint64_t>& stateIds,
    const std::vector<gtsam::Pose3>& T_W,
    const std::vector<okvis::kinematics::Transformation>& T_SC,
    const std::shared_ptr<const CameraGeometry>& camera) {
  gtsam::NonlinearFactorGraph visual;
  gtsam::Values values;
  for (std::size_t c = 0; c < T_SC.size(); ++c)
    values.insert(gb::extrinsicsKey(c), gb::toPose3(T_SC[c]));
  for (std::size_t k = 0; k < stateIds.size(); ++k)
    values.insert(gb::poseKey(stateIds[k]), T_W[k]);

  visual.addPrior(gb::poseKey(stateIds[0]), T_W[0],
                  gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));
  for (std::size_t c = 0; c < T_SC.size(); ++c)
    visual.addPrior(gb::extrinsicsKey(c), gb::toPose3(T_SC[c]),
                    gtsam::noiseModel::Isotropic::Sigma(6, 1e-6));

  std::vector<Eigen::Vector4d> lm;
  for (int ix = -2; ix <= 2; ++ix)
    for (int iy = -1; iy <= 1; ++iy)
      for (int iz = 0; iz < 2; ++iz) {
        Eigen::Vector4d p_C;
        p_C << 0.3 * ix, 0.3 * iy, 3.0 + iz, 1.0;
        lm.push_back((T_W[0].matrix() * T_SC[0].T() * p_C).eval());
      }
  const Eigen::Matrix2d info = (Eigen::Matrix2d() << 64.0, 0, 0, 64.0).finished();
  for (std::size_t l = 0; l < lm.size(); ++l) {
    values.insert(gb::landmarkKey(l), gb::toPoint3(lm[l]));
    for (std::size_t k = 0; k < stateIds.size(); ++k)
      for (std::size_t c = 0; c < T_SC.size(); ++c) {
        const Eigen::Vector4d hp_C =
            (T_SC[c].inverse().T() * T_W[k].inverse().matrix() * lm[l]);
        Eigen::Vector2d kp;
        if (camera->projectHomogeneous(hp_C, &kp) ==
            okvis::cameras::ProjectionStatus::Successful)
          visual.emplace_shared<gb::GtsamReprojectionFactor<CameraGeometry>>(
              gb::GtsamReprojectionFactor<CameraGeometry>::makeNoiseModel(info),
              gb::poseKey(stateIds[k]), gb::landmarkKey(l), gb::extrinsicsKey(c), kp,
              camera);
      }
  }
  gtsam::KeyVector drop;
  for (std::size_t l = 0; l < lm.size(); ++l) drop.push_back(gb::landmarkKey(l));
  for (std::size_t c = 0; c < T_SC.size(); ++c) drop.push_back(gb::extrinsicsKey(c));
  return gb::marginalizeOut(visual, values, drop);
}

okvis::ImuMeasurementDeque makeExcitedImu(double g, double rate, double duration) {
  okvis::ImuMeasurementDeque d;
  const double dt = 1.0 / rate;
  const int n = static_cast<int>(duration * rate) + 2;
  for (int i = 0; i < n; ++i) {
    const double t = i * dt;
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(t);
    m.measurement.gyroscopes =
        Eigen::Vector3d(0.3 * std::sin(2 * t), -0.25 * std::cos(1.5 * t), 0.2 * std::sin(t));
    m.measurement.accelerometers =
        Eigen::Vector3d(1.0 * std::sin(t), 0.8 * std::cos(2 * t), g);
    d.push_back(m);
  }
  return d;
}

}  // namespace

TEST(Pgba, RecoversGravityWithFreePosesUnderVisualPrior) {
  const okvis::ImuParameters params = makeImuParameters();
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());
  okvis::ImuMeasurementDeque imu = makeExcitedImu(params.g, 200.0, 1.2);

  // --- ground-truth trajectory in the gravity-aligned frame G --------------
  const int nFrames = 6;
  const double dt = 0.1;
  std::vector<okvis::kinematics::Transformation> T_G;
  std::vector<Eigen::Vector3d> v_G;
  okvis::kinematics::Transformation T;
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = Eigen::Vector3d(0.4, 0.1, 0.0);
  T_G.push_back(T);
  v_G.push_back(sb.head<3>());
  for (int k = 1; k < nFrames; ++k) {
    okvis::ceres::ImuError::propagation(imu, params, T, sb, okvis::Time((k - 1) * dt),
                                        okvis::Time(k * dt));
    T_G.push_back(T);
    v_G.push_back(sb.head<3>());
  }

  // Tilt into the visual world frame W by R_align (= ground-truth R_wg).
  const gtsam::Rot3 R_align = gtsam::Rot3::RzRyRx(0.0, 0.12, 0.18);
  std::vector<gtsam::Pose3> T_W(nFrames);
  std::vector<Eigen::Vector3d> v_W(nFrames);
  for (int k = 0; k < nFrames; ++k) {
    const gtsam::Rot3 Rg(Eigen::Quaterniond(T_G[k].q()));
    T_W[k] = gtsam::Pose3(R_align * Rg, R_align.rotate(T_G[k].r()));
    v_W[k] = R_align.rotate(v_G[k]);
  }

  // Stereo extrinsics.
  std::vector<okvis::kinematics::Transformation> T_SC(2);
  T_SC[0] = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Quaterniond::Identity());
  T_SC[1] = okvis::kinematics::Transformation(Eigen::Vector3d(0.11, 0, 0),
                                              Eigen::Quaterniond::Identity());

  // Landmarks in the visual world (in front of the cameras).
  std::vector<Eigen::Vector4d> lm;
  for (int ix = -2; ix <= 2; ++ix)
    for (int iy = -1; iy <= 1; ++iy)
      for (int iz = 0; iz < 2; ++iz) {
        Eigen::Vector4d p_C;
        p_C << 0.3 * ix, 0.3 * iy, 3.0 + iz, 1.0;  // camera frame of frame 0
        lm.push_back((T_W[0].matrix() * T_SC[0].T() * p_C).eval());
      }

  // --- build the stereo visual graph at ground truth, then marginalize the
  //     landmarks (and extrinsics) to obtain a pose-graph prior -------------
  gtsam::NonlinearFactorGraph visual;
  gtsam::Values visualValues;
  for (std::size_t c = 0; c < T_SC.size(); ++c)
    visualValues.insert(gb::extrinsicsKey(c), gb::toPose3(T_SC[c]));
  for (int k = 0; k < nFrames; ++k) visualValues.insert(gb::poseKey(k), T_W[k]);
  for (std::size_t l = 0; l < lm.size(); ++l)
    visualValues.insert(gb::landmarkKey(l), gb::toPoint3(lm[l]));

  // Anchor pose 0 and extrinsics so the marginal pose prior is gauge-fixed.
  visual.addPrior(gb::poseKey(0), T_W[0],
                  gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));
  for (std::size_t c = 0; c < T_SC.size(); ++c)
    visual.addPrior(gb::extrinsicsKey(c), gb::toPose3(T_SC[c]),
                    gtsam::noiseModel::Isotropic::Sigma(6, 1e-6));

  const Eigen::Matrix2d information =
      (Eigen::Matrix2d() << 64.0, 0.0, 0.0, 64.0).finished();
  int obs = 0;
  for (std::size_t l = 0; l < lm.size(); ++l)
    for (int k = 0; k < nFrames; ++k)
      for (std::size_t c = 0; c < T_SC.size(); ++c) {
        const Eigen::Vector4d hp_C =
            (T_SC[c].inverse().T() * T_W[k].inverse().matrix() * lm[l]);
        Eigen::Vector2d kp;
        if (camera->projectHomogeneous(hp_C, &kp) ==
            okvis::cameras::ProjectionStatus::Successful) {
          visual.emplace_shared<gb::GtsamReprojectionFactor<CameraGeometry>>(
              gb::GtsamReprojectionFactor<CameraGeometry>::makeNoiseModel(information),
              gb::poseKey(k), gb::landmarkKey(l), gb::extrinsicsKey(c), kp, camera);
          ++obs;
        }
      }
  ASSERT_GT(obs, 40);

  // Marginalize landmarks + extrinsics -> visual pose-graph prior over poses.
  gtsam::KeyVector toMarginalize;
  for (std::size_t l = 0; l < lm.size(); ++l) toMarginalize.push_back(gb::landmarkKey(l));
  for (std::size_t c = 0; c < T_SC.size(); ++c) toMarginalize.push_back(gb::extrinsicsKey(c));
  const gtsam::NonlinearFactor::shared_ptr visualPrior =
      gb::marginalizeOut(visual, visualValues, toMarginalize);
  ASSERT_TRUE(visualPrior != nullptr);

  // --- PGBA: free poses + velocities + gravity + bias under (visualPrior + IMU)
  gtsam::NonlinearFactorGraph pgba;
  pgba.push_back(visualPrior);

  const gtsam::Key gravKey = gtsam::Symbol('a', 0);
  const gtsam::imuBias::ConstantBias biasTrue;
  pgba.addPrior(gravKey, R_align, gtsam::noiseModel::Isotropic::Sigma(3, 1.0));  // weak gauge
  pgba.addPrior(B(0), biasTrue, gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));   // pin bias

  auto velKey = [](int k) { return gtsam::Symbol('v', k); };
  for (int k = 0; k < nFrames - 1; ++k) {
    const auto pim = gb::preintegrate(imu, params, biasTrue, okvis::Time(k * dt),
                                      okvis::Time((k + 1) * dt));
    pgba.emplace_shared<gb::PgbaImuFactor>(
        gb::PgbaImuFactor::makeNoiseModel(pim), gravKey, gb::poseKey(k), velKey(k),
        gb::poseKey(k + 1), velKey(k + 1), B(0), pim, params.g);
  }

  // Perturbed initial values: poses, velocities, gravity all off truth.
  gtsam::Values values;
  values.insert(gravKey, R_align * gtsam::Rot3::RzRyRx(0.1, -0.1, 0.08));
  values.insert(B(0), biasTrue);
  for (int k = 0; k < nFrames; ++k) {
    const gtsam::Pose3 perturbed =
        T_W[k].retract((gtsam::Vector6() << 0.02, -0.02, 0.01, 0.03, -0.02, 0.02).finished());
    values.insert(gb::poseKey(k), perturbed);
    values.insert(velKey(k), Eigen::Vector3d(v_W[k] + Eigen::Vector3d(0.2, -0.1, 0.15)));
  }

  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.setMaxIterations(100);
  gtsam::LevenbergMarquardtOptimizer optimizer(pgba, values, lmParams);
  const gtsam::Values result = optimizer.optimize();

  // Gravity direction recovered (yaw-invariant).
  const gtsam::Rot3 R_wg = result.at<gtsam::Rot3>(gravKey);
  const Eigen::Vector3d gW_est = R_wg.rotate(Eigen::Vector3d(0, 0, -params.g));
  const Eigen::Vector3d gW_true = R_align.rotate(Eigen::Vector3d(0, 0, -params.g));
  const double angle = std::acos(std::min(1.0, gW_est.normalized().dot(gW_true.normalized())));
  EXPECT_LT(angle, 0.05) << "gravity dir off by " << angle << " rad";

  // FREE poses refined back to truth under the visual prior + IMU.
  for (int k = 0; k < nFrames; ++k) {
    const gtsam::Pose3 P = result.at<gtsam::Pose3>(gb::poseKey(k));
    EXPECT_LT((P.translation() - T_W[k].translation()).norm(), 2e-2) << "pose " << k;
  }
  // Velocities recovered.
  for (int k = 0; k < nFrames; ++k)
    EXPECT_LT((result.at<Eigen::Vector3d>(velKey(k)) - v_W[k]).norm(), 0.1) << "vel " << k;
}

// Drive true PGBA through the ViImuInitializer API: when a visual marginalization
// prior is supplied, the initializer runs PGBA (free poses) rather than the
// inertial-only fallback, and recovers gravity.
TEST(Pgba, ThroughInitializerWithVisualPrior) {
  okvis::ImuParameters params = makeImuParameters();
  params.initStrategy = okvis::ImuParameters::InitStrategy::Dynamic;
  params.jointInitWindow = 6;
  params.excitationThreshAcc = 0.1;  // short window -> modest accel variance
  params.excitationThreshGyr = 0.02;
  params.initMinCondition = 1e-4;

  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());
  okvis::ImuMeasurementDeque imu = makeExcitedImu(params.g, 200.0, 0.8);

  const int nFrames = 6;
  const double dt = 0.1;
  std::vector<okvis::kinematics::Transformation> T_G;
  std::vector<okvis::Time> times;
  okvis::kinematics::Transformation T;
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = Eigen::Vector3d(1.2, 0.3, 0.0);  // enough translation to excite
  T_G.push_back(T);
  times.push_back(okvis::Time(0.0));
  for (int k = 1; k < nFrames; ++k) {
    okvis::ceres::ImuError::propagation(imu, params, T, sb, okvis::Time((k - 1) * dt),
                                        okvis::Time(k * dt));
    T_G.push_back(T);
    times.push_back(okvis::Time(k * dt));
  }

  const gtsam::Rot3 R_align = gtsam::Rot3::RzRyRx(0.0, 0.12, 0.18);
  std::vector<gtsam::Pose3> T_W(nFrames);
  std::vector<std::uint64_t> ids(nFrames);
  for (int k = 0; k < nFrames; ++k) {
    const gtsam::Rot3 Rg(Eigen::Quaterniond(T_G[k].q()));
    T_W[k] = gtsam::Pose3(R_align * Rg, R_align.rotate(T_G[k].r()));
    ids[k] = static_cast<std::uint64_t>(k + 1);
  }

  std::vector<okvis::kinematics::Transformation> T_SC(2);
  T_SC[0] = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Quaterniond::Identity());
  T_SC[1] = okvis::kinematics::Transformation(Eigen::Vector3d(0.11, 0, 0),
                                              Eigen::Quaterniond::Identity());

  const gtsam::NonlinearFactor::shared_ptr visualPrior =
      buildVisualPosePrior(ids, T_W, T_SC, camera);
  ASSERT_TRUE(visualPrior != nullptr);

  okvis::ViImuInitializer init(params);
  init.setVisualMarginalizationPrior(visualPrior);

  okvis::ViImuInitializer::Result result;
  for (int k = 0; k < nFrames; ++k) {
    const okvis::Time tPrev = (k == 0) ? times[0] : times[k - 1];
    init.addKeyframe(okvis::StateId(ids[k]),
                     okvis::kinematics::Transformation(T_W[k].translation(),
                                                       T_W[k].rotation().toQuaternion()),
                     imu, tPrev, times[k]);
    const auto r = init.step();
    if (r.converged) {
      result = r;
      break;
    }
  }

  ASSERT_TRUE(result.converged);
  EXPECT_TRUE(result.usedPgba) << "initializer should have run true PGBA";
  EXPECT_FALSE(result.usedStaticFallback);

  const Eigen::Vector3d gW_est = result.R_wg.rotate(Eigen::Vector3d(0, 0, -params.g));
  const Eigen::Vector3d gW_true = R_align.rotate(Eigen::Vector3d(0, 0, -params.g));
  const double angle = std::acos(std::min(1.0, gW_est.normalized().dot(gW_true.normalized())));
  EXPECT_LT(angle, 0.05) << "gravity dir off by " << angle << " rad";
  EXPECT_GT(static_cast<int>(result.poses.size()), 2) << "PGBA should refine poses";
}
