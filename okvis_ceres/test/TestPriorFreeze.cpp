/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *
 *  Tests for the static prior-map localization layer (Phase 1 of the lifelong
 *  mapping architecture): landmarks fixed via freezeLandmarksUntil must be
 *  immutable under optimisation, updateLandmarks and cleanUnobservedLandmarks.
 *********************************************************************************/

#include <gtest/gtest.h>
#include <okvis/ViGraphEstimator.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/assert_macros.hpp>

TEST(okvisTestSuite, PriorLandmarkFreeze) {
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  const double DURATION = 2.0;
  const double IMU_RATE = 100.0;
  const double DT = 1.0 / IMU_RATE;

  okvis::ImuParameters imuParameters;
  imuParameters.use = true;
  imuParameters.a0.setZero();
  imuParameters.g0.setZero();
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.sigma_bg = 0.01;
  imuParameters.sigma_ba = 0.01;

  // constant-velocity motion in x
  okvis::SpeedAndBias speedAndBias;
  speedAndBias.setZero();
  speedAndBias.head<3>() = Eigen::Vector3d(1, 0, 0);
  okvis::ImuMeasurementDeque imuMeasurements;
  okvis::ImuSensorReadings nominalImuSensorReadings(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, imuParameters.g));
  okvis::Time t0(100.0);
  for (size_t i = 0; i <= DURATION * IMU_RATE; ++i) {
    Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes
        + Eigen::Vector3d::Random() * imuParameters.sigma_g_c * sqrt(DT);
    Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers
        + Eigen::Vector3d::Random() * imuParameters.sigma_a_c * sqrt(DT);
    imuMeasurements.push_back(okvis::ImuMeasurement(
        t0 + okvis::Duration(DT * i), okvis::ImuSensorReadings(gyr, acc)));
  }

  // single camera, y-facing like TestViGraph2
  std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
      new okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0.0, 0),
          Eigen::Quaterniond(-sqrt(0.5), 0, 0, sqrt(0.5))));
  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0(
      okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(new okvis::cameras::NCameraSystem);
  cameraSystem->addCamera(T_SC_0, cameraGeometry0,
                          okvis::cameras::NCameraSystem::DistortionType::Equidistant);

  okvis::ViGraphEstimator graph;

  // landmark wall at y = 3
  const okvis::kinematics::Transformation T_WS_0;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> homogeneousPoints;
  std::vector<okvis::LandmarkId> lmIds;
  for (double x = -2.0; x <= DURATION * speedAndBias[0] + 2.0; x += 0.5) {
    for (double z = -2.0; z <= 2.0; z += 0.5) {
      homogeneousPoints.push_back(Eigen::Vector4d(x, 3.0, z, 1));
      lmIds.push_back(graph.addLandmark(homogeneousPoints.back(), true));
    }
  }

  okvis::CameraParameters cameraParameters; // extrinsics fixed (no online calibration)
  cameraParameters.timestamp_tolerance = 0.005;
  cameraParameters.image_delay = 0.0;
  cameraParameters.online_calibration.do_extrinsics = false;
  cameraParameters.online_calibration.do_extrinsics_final_ba = false;
  cameraParameters.online_calibration.sigma_r = 0.0;
  cameraParameters.online_calibration.sigma_alpha = 0.0;
  cameraParameters.online_calibration.sigma_r_final_ba = 0.0;
  cameraParameters.online_calibration.sigma_alpha_final_ba = 0.0;
  graph.addCamera(cameraParameters);
  graph.addImu(imuParameters);

  // 3 states with observations; lmIds[0] deliberately gets NO observations.
  const size_t K = 2;
  for (size_t k = 0; k < K + 1; ++k) {
    okvis::kinematics::Transformation T_WS(
        T_WS_0.r() + speedAndBias.head<3>() * double(k) * DURATION / double(K), T_WS_0.q());
    std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
    mf->setTimestamp(t0 + okvis::Duration(double(k) * DURATION / double(K)));
    mf->resetCameraSystemAndFrames(*cameraSystem);
    okvis::StateId id;
    if (k == 0) {
      id = graph.addStatesInitialise(mf->timestamp(), imuMeasurements, *cameraSystem);
    } else {
      id = graph.addStatesPropagate(mf->timestamp(), imuMeasurements, true);
    }
    mf->setId(id.value());
    std::vector<cv::KeyPoint> keypoints;
    for (size_t j = 1; j < homogeneousPoints.size(); ++j) {
      Eigen::Vector2d projection;
      Eigen::Vector4d point_C = mf->T_SC(0)->inverse() * T_WS.inverse() * homogeneousPoints[j];
      auto status = mf->geometryAs<okvis::cameras::PinholeCamera<
          okvis::cameras::EquidistantDistortion>>(0)->projectHomogeneous(point_C, &projection);
      if (status == okvis::cameras::ProjectionStatus::Successful) {
        Eigen::Vector2d measurement(projection + Eigen::Vector2d::Random());
        keypoints.push_back(cv::KeyPoint(static_cast<float>(measurement[0]),
                                         static_cast<float>(measurement[1]), 8.0));
        mf->resetKeypoints(0, keypoints);
        okvis::KeypointIdentifier kid(mf->id(), 0, mf->numKeypoints(0) - 1);
        graph.addObservation<okvis::cameras::PinholeCamera<
            okvis::cameras::EquidistantDistortion>>(*mf, lmIds[j], kid);
      }
    }
  }

  // an unfrozen landmark with zero observations -- must be culled
  const okvis::LandmarkId unobservedLiveId =
      graph.addLandmark(Eigen::Vector4d(0.0, -5.0, 0.0, 1.0), false);

  // treat the first half of the landmarks as the "prior map": freeze them
  const size_t numPrior = lmIds.size() / 2;
  const okvis::LandmarkId priorMaxLandmarkId = lmIds[numPrior - 1];
  const int numFrozen = graph.freezeLandmarksUntil(priorMaxLandmarkId);
  EXPECT_EQ(numFrozen, int(numPrior));
  // idempotent
  EXPECT_EQ(graph.freezeLandmarksUntil(priorMaxLandmarkId), 0);

  // snapshot the frozen landmarks
  std::map<uint64_t, okvis::MapPoint2> before;
  for (size_t j = 0; j < numPrior; ++j) {
    okvis::MapPoint2 mp;
    EXPECT_TRUE(graph.getLandmark(lmIds[j], mp));
    before[lmIds[j].value()] = mp;
  }

  // (i) optimisation must not move frozen landmarks
  graph.optimise(10, 1, false);
  // (ii) updateLandmarks must not touch estimate/quality/init of frozen landmarks
  graph.updateLandmarks();
  bool anyUnfrozenQualityUpdated = false;
  for (size_t j = 0; j < lmIds.size(); ++j) {
    okvis::MapPoint2 mp;
    EXPECT_TRUE(graph.getLandmark(lmIds[j], mp));
    if (j < numPrior) {
      const okvis::MapPoint2 &ref = before[lmIds[j].value()];
      EXPECT_TRUE((mp.point - ref.point).norm() == 0.0)
          << "frozen landmark " << lmIds[j].value() << " moved";
      EXPECT_EQ(mp.isInitialised, ref.isInitialised);
      EXPECT_EQ(mp.quality, ref.quality);
    } else if (mp.quality > 0.0) {
      anyUnfrozenQualityUpdated = true;
    }
  }
  EXPECT_TRUE(anyUnfrozenQualityUpdated) << "updateLandmarks did not run on live landmarks";

  // (iii) cleanUnobservedLandmarks spares frozen landmarks (incl. zero-observation
  // ones) and removes unfrozen zero-observation ones
  graph.cleanUnobservedLandmarks();
  okvis::MapPoint2 mp;
  EXPECT_TRUE(graph.getLandmark(lmIds[0], mp)) << "frozen zero-obs landmark was culled";
  EXPECT_EQ(mp.observations.size(), 0u);
  EXPECT_FALSE(graph.landmarkExists(unobservedLiveId))
      << "unfrozen zero-obs landmark was not culled";
}
