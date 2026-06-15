/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestGtsamReprojection.cpp
 * @brief Validates GtsamReprojectionFactor: zero error at ground truth, analytic
 *        Jacobians vs numericalDerivative, and residual parity with the Ceres
 *        okvis::ceres::ReprojectionError.
 */

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/ReprojectionError.hpp>

#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/GtsamReprojectionFactor.hpp>

namespace {
typedef okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>
    CameraGeometry;
typedef okvis::gtsam_backend::GtsamReprojectionFactor<CameraGeometry> Factor;
}  // namespace

class GtsamReprojection : public ::testing::Test {
 protected:
  void SetUp() override {
    camera_ = std::static_pointer_cast<const CameraGeometry>(
        CameraGeometry::createTestObject());
    // Non-trivial body pose and extrinsics.
    T_WS_ = okvis::kinematics::Transformation(
        Eigen::Vector3d(0.3, -0.5, 1.2),
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.25, Eigen::Vector3d(0.2, 1.0, -0.3).normalized())));
    T_SC_ = okvis::kinematics::Transformation(
        Eigen::Vector3d(0.05, 0.02, -0.01),
        Eigen::Quaterniond(
            Eigen::AngleAxisd(0.05, Eigen::Vector3d(1.0, 0.1, 0.0).normalized())));

    // A visible point in the camera frame, lifted to the world frame.
    const Eigen::Vector4d hp_C = camera_->createRandomVisibleHomogeneousPoint(3.0);
    hp_W_ = (T_WS_.T() * T_SC_.T() * hp_C).eval();

    // The "true" measurement = projection at ground truth.
    Eigen::Vector4d hp_C_check = (T_SC_.inverse().T() * T_WS_.inverse().T() * hp_W_);
    camera_->projectHomogeneous(hp_C_check, &measurement_);
  }

  std::shared_ptr<const CameraGeometry> camera_;
  okvis::kinematics::Transformation T_WS_;
  okvis::kinematics::Transformation T_SC_;
  Eigen::Vector4d hp_W_;
  Eigen::Vector2d measurement_;
};

TEST_F(GtsamReprojection, ZeroErrorAtGroundTruth) {
  ASSERT_TRUE(camera_ != nullptr);
  const Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  Factor factor(Factor::makeNoiseModel(information), okvis::gtsam_backend::poseKey(0),
                okvis::gtsam_backend::landmarkKey(0),
                okvis::gtsam_backend::extrinsicsKey(0), measurement_, camera_);

  const gtsam::Vector e = factor.evaluateError(
      okvis::gtsam_backend::toPose3(T_WS_),
      okvis::gtsam_backend::toPoint3(hp_W_),
      okvis::gtsam_backend::toPose3(T_SC_));
  EXPECT_LT(e.norm(), 1e-8) << "error: " << e.transpose();
}

TEST_F(GtsamReprojection, JacobiansMatchNumerical) {
  ASSERT_TRUE(camera_ != nullptr);
  Factor factor(Factor::makeNoiseModel(Eigen::Matrix2d::Identity()),
                okvis::gtsam_backend::poseKey(0),
                okvis::gtsam_backend::landmarkKey(0),
                okvis::gtsam_backend::extrinsicsKey(0), measurement_, camera_);

  // Evaluate at perturbed states so the error (and Jacobians) are non-degenerate.
  const gtsam::Pose3 T_WS = okvis::gtsam_backend::toPose3(T_WS_)
      .retract((gtsam::Vector6() << 0.02, -0.03, 0.01, 0.05, -0.04, 0.02).finished());
  const gtsam::Point3 p_W =
      okvis::gtsam_backend::toPoint3(hp_W_) + gtsam::Point3(0.05, -0.03, 0.04);
  const gtsam::Pose3 T_SC = okvis::gtsam_backend::toPose3(T_SC_);

  gtsam::Matrix H_WS, H_pW, H_SC;
  factor.evaluateError(T_WS, p_W, T_SC, H_WS, H_pW, H_SC);

  const auto fPose = [&](const gtsam::Pose3& x) {
    return factor.evaluateError(x, p_W, T_SC);
  };
  const auto fPoint = [&](const gtsam::Point3& x) {
    return factor.evaluateError(T_WS, x, T_SC);
  };
  const auto fExtr = [&](const gtsam::Pose3& x) {
    return factor.evaluateError(T_WS, p_W, x);
  };

  const gtsam::Matrix H_WS_num =
      gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(fPose, T_WS, 1e-6);
  const gtsam::Matrix H_pW_num =
      gtsam::numericalDerivative11<gtsam::Vector, gtsam::Point3>(fPoint, p_W, 1e-6);
  const gtsam::Matrix H_SC_num =
      gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(fExtr, T_SC, 1e-6);

  EXPECT_LT((H_WS - H_WS_num).norm(), 1e-4) << "H_WS:\n" << H_WS << "\nnum:\n" << H_WS_num;
  EXPECT_LT((H_pW - H_pW_num).norm(), 1e-4) << "H_pW:\n" << H_pW << "\nnum:\n" << H_pW_num;
  EXPECT_LT((H_SC - H_SC_num).norm(), 1e-4) << "H_SC:\n" << H_SC << "\nnum:\n" << H_SC_num;
}

TEST_F(GtsamReprojection, ResidualParityWithCeres) {
  ASSERT_TRUE(camera_ != nullptr);
  const Eigen::Matrix2d information = (Eigen::Matrix2d() << 64.0, 0.0, 0.0, 64.0).finished();

  Factor factor(Factor::makeNoiseModel(information), okvis::gtsam_backend::poseKey(0),
                okvis::gtsam_backend::landmarkKey(0),
                okvis::gtsam_backend::extrinsicsKey(0), measurement_, camera_);

  okvis::ceres::ReprojectionError<CameraGeometry> ceresError(camera_, 0, measurement_,
                                                             information);

  // Perturb the pose so the residual is non-zero.
  const gtsam::Vector6 dx =
      (gtsam::Vector6() << 0.01, 0.0, -0.02, 0.03, 0.01, -0.01).finished();
  const gtsam::Pose3 T_WS_p = okvis::gtsam_backend::toPose3(T_WS_).retract(dx);
  const okvis::kinematics::Transformation T_WS_pert =
      okvis::gtsam_backend::fromPose3(T_WS_p);

  // GTSAM unweighted error.
  const gtsam::Vector e_gtsam = factor.evaluateError(
      T_WS_p, okvis::gtsam_backend::toPoint3(hp_W_),
      okvis::gtsam_backend::toPose3(T_SC_));

  // Ceres weighted residual: sqrtInfo * (measurement - prediction).
  double residuals[2];
  const double* params[3];
  // OKVIS pose parameter layout: [t(3), q_x, q_y, q_z, q_w]. Build manually to
  // avoid Transformation::parameters() which is disallowed in cached mode.
  Eigen::Matrix<double, 7, 1> poseParam;
  poseParam.head<3>() = T_WS_pert.r();
  poseParam.segment<4>(3) = Eigen::Quaterniond(T_WS_pert.q()).coeffs();
  Eigen::Vector4d pointParam = hp_W_;
  Eigen::Matrix<double, 7, 1> extrParam;
  extrParam.head<3>() = T_SC_.r();
  extrParam.segment<4>(3) = Eigen::Quaterniond(T_SC_.q()).coeffs();
  params[0] = poseParam.data();
  params[1] = pointParam.data();
  params[2] = extrParam.data();
  ceresError.Evaluate(params, residuals, nullptr);

  // sqrtInfo = 8*I, gtsam error = prediction - measurement = -(measurement - pred).
  const Eigen::Vector2d ceresRes(residuals[0], residuals[1]);
  const Eigen::Vector2d expected = -8.0 * e_gtsam;  // sqrt(64) = 8
  EXPECT_LT((ceresRes - expected).norm(), 1e-6)
      << "ceres: " << ceresRes.transpose() << "  expected: " << expected.transpose();
}
