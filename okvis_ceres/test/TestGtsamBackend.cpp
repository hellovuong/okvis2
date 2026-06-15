/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGtsamBackend.cpp
 * @brief End-to-end test of GtsamBackend on a synthetic stereo + IMU trajectory:
 *        builds the factor graph, perturbs the initial estimate, and checks that
 *        batch LM recovers the ground-truth poses, velocities and landmarks.
 *
 * Motion: constant world velocity v=(1,0,0) m/s, no rotation. For a level static
 * orientation the accelerometer specific force is (0,0,+g) and the gyro is zero.
 */

#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

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

TEST(GtsamBackend, StereoImuConvergence) {
  const okvis::ImuParameters imuParameters = makeImuParameters();
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());

  // --- ground-truth trajectory: constant velocity, level, no rotation -------
  const int nFrames = 4;
  const double dt = 0.1;
  const Eigen::Vector3d v_W(1.0, 0.0, 0.0);
  std::vector<okvis::kinematics::Transformation> T_WS_gt(nFrames);
  for (int k = 0; k < nFrames; ++k) {
    T_WS_gt[k] = okvis::kinematics::Transformation(v_W * (k * dt),
                                                   Eigen::Quaterniond::Identity());
  }
  okvis::SpeedAndBias sb_gt;
  sb_gt.head<3>() = v_W;
  sb_gt.segment<3>(3).setZero();  // gyro bias
  sb_gt.tail<3>().setZero();      // accel bias

  // Stereo extrinsics: two forward-looking cameras with a 0.11 m baseline.
  std::vector<okvis::kinematics::Transformation> T_SC(2);
  T_SC[0] = okvis::kinematics::Transformation(Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Quaterniond::Identity());
  T_SC[1] = okvis::kinematics::Transformation(Eigen::Vector3d(0.11, 0.0, 0.0),
                                              Eigen::Quaterniond::Identity());

  // --- landmarks in front of the cameras ------------------------------------
  std::vector<Eigen::Vector4d> lm_gt;
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int iz = 0; iz < 2; ++iz) {
        Eigen::Vector4d p;
        p << 0.4 * ix, 0.4 * iy, 3.0 + iz, 1.0;  // z forward (camera looks +z)
        lm_gt.push_back(p);
      }
    }
  }
  const int nLandmarks = static_cast<int>(lm_gt.size());

  // --- IMU stream consistent with the motion --------------------------------
  // Level, static orientation: accelerometer reads (0,0,+g); gyro = 0.
  okvis::ImuMeasurementDeque imu;
  for (int i = 0; i < static_cast<int>((nFrames - 1) * dt / 0.005) + 4; ++i) {
    okvis::ImuMeasurement m;
    m.timeStamp = okvis::Time(i * 0.005);
    m.measurement.gyroscopes = Eigen::Vector3d::Zero();
    m.measurement.accelerometers = Eigen::Vector3d(0.0, 0.0, imuParameters.g);
    imu.push_back(m);
  }

  // --- build the backend ----------------------------------------------------
  okvis::GtsamBackend backend(imuParameters);
  for (std::size_t c = 0; c < T_SC.size(); ++c) {
    backend.setExtrinsics(c, T_SC[c], /*fixed=*/true);
  }

  // Initial values: perturb everything except gauge-fixed first pose.
  auto perturbPose = [](const okvis::kinematics::Transformation& T, double tmag,
                        double rmag) {
    return okvis::kinematics::Transformation(
        T.r() + Eigen::Vector3d(tmag, -tmag, tmag),
        T.q() * Eigen::Quaterniond(Eigen::AngleAxisd(
                    rmag, Eigen::Vector3d(0.2, 1.0, -0.3).normalized())));
  };

  for (int k = 0; k < nFrames; ++k) {
    okvis::SpeedAndBias sb_init = sb_gt;
    sb_init.head<3>() += Eigen::Vector3d(0.2, -0.15, 0.1);  // velocity perturbation
    const okvis::kinematics::Transformation T_init =
        (k == 0) ? T_WS_gt[k] : perturbPose(T_WS_gt[k], 0.10, 0.05);
    backend.addState(okvis::StateId(k + 1), T_init, sb_init);
  }
  // Gauge: anchor first pose tightly, plus a moderate velocity/bias prior.
  backend.addPosePrior(okvis::StateId(1), T_WS_gt[0], 1e-4, 1e-4);
  backend.addSpeedAndBiasPrior(okvis::StateId(1), sb_gt, 0.2, 0.01, 0.05);

  // IMU factors between consecutive frames.
  for (int k = 0; k < nFrames - 1; ++k) {
    backend.addImuFactor(okvis::StateId(k + 1), okvis::StateId(k + 2), imu,
                         okvis::Time(k * dt), okvis::Time((k + 1) * dt));
  }

  // Landmarks (perturbed) + stereo observations at every frame.
  const Eigen::Matrix2d information = (Eigen::Matrix2d() << 64.0, 0.0, 0.0,
                                       64.0).finished();
  for (int l = 0; l < nLandmarks; ++l) {
    Eigen::Vector4d lm_init = lm_gt[l];
    lm_init.head<3>() += Eigen::Vector3d(0.05, -0.08, 0.1);
    backend.addLandmark(okvis::LandmarkId(1000 + l), lm_init);

    for (int k = 0; k < nFrames; ++k) {
      for (std::size_t c = 0; c < T_SC.size(); ++c) {
        const Eigen::Vector4d hp_C =
            (T_SC[c].inverse().T() * T_WS_gt[k].inverse().T() * lm_gt[l]);
        Eigen::Vector2d kp;
        if (camera->projectHomogeneous(hp_C, &kp) ==
            okvis::cameras::ProjectionStatus::Successful) {
          backend.addObservation<CameraGeometry>(okvis::StateId(k + 1),
                                                  okvis::LandmarkId(1000 + l), c,
                                                  kp, information, camera);
        }
      }
    }
  }

  const double errorBefore = backend.error();
  const double errorAfter = backend.optimise(30);
  EXPECT_LT(errorAfter, errorBefore);

  // --- check convergence to ground truth ------------------------------------
  for (int k = 0; k < nFrames; ++k) {
    const okvis::kinematics::Transformation T = backend.getPose(okvis::StateId(k + 1));
    EXPECT_LT((T.r() - T_WS_gt[k].r()).norm(), 5e-3)
        << "frame " << k << " position off";
    EXPECT_LT(Eigen::Quaterniond(T.q()).angularDistance(
                  Eigen::Quaterniond(T_WS_gt[k].q())), 5e-3)
        << "frame " << k << " orientation off";

    const okvis::SpeedAndBias sb = backend.getSpeedAndBias(okvis::StateId(k + 1));
    EXPECT_LT((sb.head<3>() - v_W).norm(), 5e-2) << "frame " << k << " velocity off";
  }
}

// Track-5 S1: Estimator-API state/window queries.
TEST(GtsamBackend, StateMetadataAndQueries) {
  okvis::GtsamBackend backend(makeImuParameters());
  const okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  const okvis::kinematics::Transformation I;

  EXPECT_EQ(backend.numFrames(), 0u);
  EXPECT_FALSE(backend.currentStateId().isInitialised());

  backend.addState(okvis::StateId(1), I, sb, okvis::Time(1.0), true);
  backend.addState(okvis::StateId(2), I, sb, okvis::Time(2.0), false);
  backend.addState(okvis::StateId(3), I, sb, okvis::Time(3.0), true);

  EXPECT_EQ(backend.numFrames(), 3u);
  EXPECT_EQ(backend.currentStateId().value(), 3u);
  EXPECT_EQ(backend.stateIdByAge(0).value(), 3u);   // newest
  EXPECT_EQ(backend.stateIdByAge(2).value(), 1u);   // oldest
  EXPECT_FALSE(backend.stateIdByAge(3).isInitialised());
  EXPECT_NEAR(backend.timestamp(okvis::StateId(2)).toSec(), 2.0, 1e-9);
  EXPECT_TRUE(backend.isKeyframe(okvis::StateId(1)));
  EXPECT_FALSE(backend.isKeyframe(okvis::StateId(2)));
  backend.setKeyframe(okvis::StateId(2), true);
  EXPECT_TRUE(backend.isKeyframe(okvis::StateId(2)));

  // Marginalizing the oldest state drops it from the window + metadata.
  backend.marginalizeState(okvis::StateId(1));
  EXPECT_EQ(backend.numFrames(), 2u);
  EXPECT_FALSE(backend.hasState(okvis::StateId(1)));
  EXPECT_FALSE(backend.isKeyframe(okvis::StateId(1)));
  EXPECT_EQ(backend.currentStateId().value(), 3u);
}

// Track-5 S1: per-frame IMU-propagated state creation (addPropagatedState).
TEST(GtsamBackend, PropagatedStateSequenceStaysConsistent) {
  const okvis::ImuParameters params = makeImuParameters();
  okvis::GtsamBackend backend(params);
  backend.setExtrinsics(0, okvis::kinematics::Transformation(), true);

  auto imuFor = [&](double t0, double t1) {
    okvis::ImuMeasurementDeque d;
    const double dt = 1.0 / 200.0;
    for (double t = t0 - 0.01; t <= t1 + 0.01 + 1e-9; t += dt) {
      okvis::ImuMeasurement m;
      m.timeStamp = okvis::Time(t < 0 ? 0.0 : t);
      m.measurement.gyroscopes = Eigen::Vector3d::Zero();
      m.measurement.accelerometers = Eigen::Vector3d(0, 0, params.g);  // static
      d.push_back(m);
    }
    return d;
  };

  const double dt = 0.1;
  const int n = 5;
  for (int k = 0; k < n; ++k) {
    const okvis::ImuMeasurementDeque imu =
        imuFor(k == 0 ? 0.0 : (k - 1) * dt, k * dt);
    EXPECT_TRUE(backend.addPropagatedState(okvis::StateId(k + 1), okvis::Time(k * dt),
                                           imu, true));
  }
  EXPECT_EQ(backend.numFrames(), static_cast<std::size_t>(n));
  EXPECT_EQ(backend.currentStateId().value(), static_cast<std::uint64_t>(n));

  const double err = backend.optimise(15);
  EXPECT_TRUE(std::isfinite(err));

  // Static body under gravity: every frame should stay near the origin.
  for (int k = 0; k < n; ++k) {
    const okvis::kinematics::Transformation T = backend.getPose(okvis::StateId(k + 1));
    EXPECT_LT(T.r().norm(), 0.05) << "frame " << k << " drifted: " << T.r().transpose();
  }
}

// Track-5 S1: sliding-window strategy marginalizes old non-kept frames.
TEST(GtsamBackend, WindowStrategyMarginalisesOldFrames) {
  const okvis::ImuParameters params = makeImuParameters();
  okvis::GtsamBackend backend(params);
  backend.setExtrinsics(0, okvis::kinematics::Transformation(), true);

  auto imuFor = [&](double t0, double t1) {
    okvis::ImuMeasurementDeque d;
    const double dt = 1.0 / 200.0;
    for (double t = t0 - 0.01; t <= t1 + 0.01 + 1e-9; t += dt) {
      okvis::ImuMeasurement m;
      m.timeStamp = okvis::Time(t < 0 ? 0.0 : t);
      m.measurement.gyroscopes = Eigen::Vector3d::Zero();
      m.measurement.accelerometers = Eigen::Vector3d(0, 0, params.g);
      d.push_back(m);
    }
    return d;
  };

  const double dt = 0.1;
  const int n = 8;
  for (int k = 0; k < n; ++k) {  // keyframes at odd ids (1,3,5,7)
    backend.addPropagatedState(okvis::StateId(k + 1), okvis::Time(k * dt),
                               imuFor(k == 0 ? 0.0 : (k - 1) * dt, k * dt),
                               /*asKeyframe=*/(k % 2 == 0));
  }
  EXPECT_EQ(backend.numFrames(), 8u);

  std::set<okvis::StateId> affected;
  backend.applyStrategy(/*numKeyframes=*/2, /*numLoopClosureFrames=*/0,
                        /*numImuFrames=*/3, affected, true);

  // keep = newest 3 {6,7,8} + newest 2 keyframes {5,7} = {5,6,7,8}.
  EXPECT_EQ(backend.numFrames(), 4u);
  for (int id : {5, 6, 7, 8}) EXPECT_TRUE(backend.hasState(okvis::StateId(id))) << id;
  for (int id : {1, 2, 3, 4}) EXPECT_FALSE(backend.hasState(okvis::StateId(id))) << id;
  EXPECT_EQ(affected.size(), 4u);
  // IMU window = newest 3.
  EXPECT_EQ(backend.imuFrames().size(), 3u);
  EXPECT_TRUE(backend.isInImuWindow(okvis::StateId(8)));
  EXPECT_FALSE(backend.isInImuWindow(okvis::StateId(5)));
  // kept keyframes are 5 and 7.
  EXPECT_TRUE(backend.keyFrames().count(okvis::StateId(5)) > 0);
  EXPECT_TRUE(backend.keyFrames().count(okvis::StateId(7)) > 0);

  std::vector<okvis::StateId> updated;
  backend.optimiseRealtimeGraph(10, updated);
  EXPECT_EQ(updated.size(), 4u);
  for (int id : {5, 6, 7, 8}) {
    EXPECT_LT(backend.getPose(okvis::StateId(id)).r().norm(), 0.05) << "kept " << id;
  }
}

// Track-5 S2: landmark metadata (init/classification/quality) + getLandmark.
TEST(GtsamBackend, LandmarkMetadata) {
  okvis::GtsamBackend backend(makeImuParameters());
  Eigen::Vector4d p;
  p << 1, 2, 3, 1;

  EXPECT_FALSE(backend.isLandmarkAdded(okvis::LandmarkId(10)));
  backend.addLandmark(okvis::LandmarkId(10), p, /*initialised=*/false);
  EXPECT_TRUE(backend.isLandmarkAdded(okvis::LandmarkId(10)));
  EXPECT_FALSE(backend.isLandmarkInitialised(okvis::LandmarkId(10)));

  EXPECT_TRUE(backend.setLandmarkInitialized(okvis::LandmarkId(10), true));
  EXPECT_TRUE(backend.isLandmarkInitialised(okvis::LandmarkId(10)));
  EXPECT_TRUE(backend.setLandmarkClassification(okvis::LandmarkId(10), 7));

  okvis::MapPoint2 mp;
  ASSERT_TRUE(backend.getLandmark(okvis::LandmarkId(10), mp));
  EXPECT_EQ(mp.id.value(), 10u);
  EXPECT_EQ(mp.classification, 7);
  EXPECT_TRUE(mp.isInitialised);
  EXPECT_LT((mp.point - p).norm(), 1e-9);

  Eigen::Vector4d p2;
  p2 << 4, 5, 6, 1;
  EXPECT_TRUE(backend.setLandmark(okvis::LandmarkId(10), p2, /*isInitialised=*/false));
  ASSERT_TRUE(backend.getLandmark(okvis::LandmarkId(10), mp));
  EXPECT_LT((mp.point - p2).norm(), 1e-9);
  EXPECT_FALSE(mp.isInitialised);

  // Unknown landmark.
  EXPECT_FALSE(backend.getLandmark(okvis::LandmarkId(99), mp));
  EXPECT_FALSE(backend.setLandmark(okvis::LandmarkId(99), p2, true));
  EXPECT_FALSE(backend.setLandmarkInitialized(okvis::LandmarkId(99), true));
}

// Track-5 S2: observation lifecycle (add/query/remove + cleanUnobservedLandmarks).
TEST(GtsamBackend, ObservationLifecycle) {
  okvis::GtsamBackend backend(makeImuParameters());
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());
  backend.setExtrinsics(0, okvis::kinematics::Transformation(), true);
  backend.addState(okvis::StateId(1), okvis::kinematics::Transformation(),
                   okvis::SpeedAndBias::Zero());
  backend.addPosePrior(okvis::StateId(1), okvis::kinematics::Transformation(), 1e-3, 1e-2);

  Eigen::Vector4d lm;
  lm << 0.2, -0.1, 3.0, 1.0;  // in front of cam0 (pose + extrinsics identity)
  backend.addLandmark(okvis::LandmarkId(100), lm, true);
  Eigen::Vector2d kp;
  ASSERT_EQ(camera->projectHomogeneous(lm, &kp),
            okvis::cameras::ProjectionStatus::Successful);
  const Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * 64.0;

  const okvis::KeypointIdentifier kid1(1, 0, 0), kid2(1, 0, 1);
  EXPECT_FALSE(backend.isObserved(kid1));
  EXPECT_TRUE(backend.addObservation<CameraGeometry>(okvis::LandmarkId(100), kid1, kp,
                                                     info, camera));
  EXPECT_TRUE(backend.isObserved(kid1));
  // duplicate rejected
  EXPECT_FALSE(backend.addObservation<CameraGeometry>(okvis::LandmarkId(100), kid1, kp,
                                                      info, camera));
  EXPECT_TRUE(backend.addObservation<CameraGeometry>(okvis::LandmarkId(100), kid2, kp,
                                                     info, camera));
  EXPECT_TRUE(std::isfinite(backend.optimise(5)));

  // Remove one observation; landmark still observed -> not cleaned.
  EXPECT_TRUE(backend.removeObservation(kid1));
  EXPECT_FALSE(backend.isObserved(kid1));
  EXPECT_TRUE(backend.isObserved(kid2));
  EXPECT_EQ(backend.cleanUnobservedLandmarks(), 0);
  EXPECT_TRUE(backend.isLandmarkAdded(okvis::LandmarkId(100)));

  // Remove the last observation -> landmark unobserved -> cleaned.
  EXPECT_TRUE(backend.removeObservation(kid2));
  EXPECT_EQ(backend.cleanUnobservedLandmarks(), 1);
  EXPECT_FALSE(backend.isLandmarkAdded(okvis::LandmarkId(100)));
  EXPECT_TRUE(std::isfinite(backend.optimise(5)));
}

// Track-5 S2: covisibility (getObservedIds) + getLandmarks; loop-closure stubs inert.
TEST(GtsamBackend, CovisibilityAndLandmarkDump) {
  okvis::GtsamBackend backend(makeImuParameters());
  auto camera = std::static_pointer_cast<const CameraGeometry>(
      CameraGeometry::createTestObject());
  backend.setExtrinsics(0, okvis::kinematics::Transformation(), true);
  for (int s = 1; s <= 3; ++s)
    backend.addState(okvis::StateId(s), okvis::kinematics::Transformation(),
                     okvis::SpeedAndBias::Zero());

  Eigen::Vector4d lmA, lmB;
  lmA << 0.1, 0.0, 3.0, 1.0;
  lmB << -0.1, 0.1, 4.0, 1.0;
  backend.addLandmark(okvis::LandmarkId(1), lmA, true);
  backend.addLandmark(okvis::LandmarkId(2), lmB, false);
  Eigen::Vector2d kpA, kpB;
  camera->projectHomogeneous(lmA, &kpA);
  camera->projectHomogeneous(lmB, &kpB);
  const Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * 64.0;

  // landmark 1 observed by states 1 & 2; landmark 2 by state 3 only.
  backend.addObservation<CameraGeometry>(okvis::LandmarkId(1),
                                         okvis::KeypointIdentifier(1, 0, 0), kpA, info, camera);
  backend.addObservation<CameraGeometry>(okvis::LandmarkId(1),
                                         okvis::KeypointIdentifier(2, 0, 0), kpA, info, camera);
  backend.addObservation<CameraGeometry>(okvis::LandmarkId(2),
                                         okvis::KeypointIdentifier(3, 0, 0), kpB, info, camera);

  std::set<okvis::StateId> coObs;
  EXPECT_TRUE(backend.getObservedIds(okvis::StateId(1), coObs));
  EXPECT_EQ(coObs.size(), 1u);                       // state 2 co-observes landmark 1
  EXPECT_TRUE(coObs.count(okvis::StateId(2)) > 0);
  EXPECT_FALSE(coObs.count(okvis::StateId(3)) > 0);  // disjoint landmark

  okvis::MapPoints dump;
  EXPECT_EQ(backend.getLandmarks(dump), 2u);
  EXPECT_TRUE(dump.at(okvis::LandmarkId(1)).isInitialised);
  EXPECT_FALSE(dump.at(okvis::LandmarkId(2)).isInitialised);

  // Loop-closure stubs are inert.
  EXPECT_FALSE(backend.isLoopClosing());
  EXPECT_FALSE(backend.isLoopClosureAvailable());
  EXPECT_FALSE(backend.needsFullGraphOptimisation());
  EXPECT_TRUE(backend.loopClosureFrames().empty());
}
