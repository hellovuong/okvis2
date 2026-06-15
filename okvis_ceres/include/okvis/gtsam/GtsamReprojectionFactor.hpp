/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file GtsamReprojectionFactor.hpp
 * @brief A GTSAM reprojection factor that reuses the OKVIS camera geometry.
 *
 * Connects pose T_WS (X), landmark in world frame (L, gtsam::Point3) and camera
 * extrinsics T_SC (E). Reproduces the projection math of
 * okvis::ceres::ReprojectionError (hp_C = T_CS * T_SW * hp_W, then the OKVIS
 * distortion model via projectHomogeneous) but expresses Jacobians in GTSAM's
 * Pose3 tangent convention ([omega, v]) by chaining GTSAM's own transformTo
 * derivatives — so we never have to port OKVIS's [t, rot] minimal Jacobians.
 *
 * The 2x2 measurement weighting is carried by the GTSAM noise model (NOT
 * pre-multiplied into the residual here); wrap it in noiseModel::Robust with a
 * Cauchy m-estimator to mirror OKVIS's robust reprojection loss.
 */

#ifndef INCLUDE_OKVIS_GTSAM_GTSAMREPROJECTIONFACTOR_HPP_
#define INCLUDE_OKVIS_GTSAM_GTSAMREPROJECTIONFACTOR_HPP_

#include <memory>

#include <Eigen/Core>

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace okvis {
namespace gtsam_backend {

/// \brief 2D keypoint reprojection factor over (T_WS, point_W, T_SC).
/// \tparam GEOMETRY_TYPE An OKVIS camera geometry type exposing
///         projectHomogeneous(const Eigen::Vector4d&, Eigen::Vector2d*,
///                             Eigen::Matrix<double,2,4>*).
template <class GEOMETRY_TYPE>
class GtsamReprojectionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Point3, gtsam::Pose3> {
 public:
  typedef GEOMETRY_TYPE camera_geometry_t;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Point3, gtsam::Pose3> Base;

  GtsamReprojectionFactor() = default;

  /// \brief Construct.
  /// \param model       The (possibly robust) 2D noise model.
  /// \param poseKey     Key of the body pose T_WS.
  /// \param landmarkKey Key of the world-frame landmark (Point3).
  /// \param extrinsicsKey Key of the camera extrinsics T_SC.
  /// \param measurement The measured keypoint.
  /// \param cameraGeometry The OKVIS camera geometry.
  GtsamReprojectionFactor(
      const gtsam::SharedNoiseModel& model, gtsam::Key poseKey,
      gtsam::Key landmarkKey, gtsam::Key extrinsicsKey,
      const Eigen::Vector2d& measurement,
      std::shared_ptr<const camera_geometry_t> cameraGeometry)
      : Base(model, poseKey, landmarkKey, extrinsicsKey),
        measurement_(measurement),
        cameraGeometry_(std::move(cameraGeometry)) {}

  ~GtsamReprojectionFactor() override = default;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new GtsamReprojectionFactor(*this));
  }

  /// \brief Reprojection error h(x) - z, with analytic Jacobians.
  gtsam::Vector evaluateError(
      const gtsam::Pose3& T_WS, const gtsam::Point3& p_W,
      const gtsam::Pose3& T_SC, boost::optional<gtsam::Matrix&> H_WS = boost::none,
      boost::optional<gtsam::Matrix&> H_pW = boost::none,
      boost::optional<gtsam::Matrix&> H_SC = boost::none) const override {
    const bool needJac = H_WS || H_pW || H_SC;

    // p_S = T_WS^{-1} * p_W ; p_C = T_SC^{-1} * p_S.
    gtsam::Matrix36 D_pS_TWS, D_pC_TSC;
    gtsam::Matrix33 D_pS_pW, D_pC_pS;
    const gtsam::Point3 p_S = needJac
        ? T_WS.transformTo(p_W, D_pS_TWS, D_pS_pW)
        : T_WS.transformTo(p_W);
    const gtsam::Point3 p_C = needJac
        ? T_SC.transformTo(p_S, D_pC_TSC, D_pC_pS)
        : T_SC.transformTo(p_S);

    Eigen::Vector4d hp_C;
    hp_C << p_C.x(), p_C.y(), p_C.z(), 1.0;

    Eigen::Vector2d kp;
    Eigen::Matrix<double, 2, 3> D_kp_pC;
    if (needJac) {
      Eigen::Matrix<double, 2, 4> Jh;
      cameraGeometry_->projectHomogeneous(hp_C, &kp, &Jh);
      D_kp_pC = Jh.leftCols<3>();  // d(kp)/d(p_C); hp_C[3]=1 is constant
    } else {
      cameraGeometry_->projectHomogeneous(hp_C, &kp);
    }

    if (H_WS) *H_WS = D_kp_pC * D_pC_pS * D_pS_TWS;  // 2x6
    if (H_pW) *H_pW = D_kp_pC * D_pC_pS * D_pS_pW;   // 2x3
    if (H_SC) *H_SC = D_kp_pC * D_pC_TSC;            // 2x6

    return kp - measurement_;
  }

  /// \brief Build a robust (Cauchy) 2D noise model from an OKVIS 2x2 information.
  /// \param information The 2x2 information (inverse covariance) matrix.
  /// \param cauchyParam The Cauchy m-estimator parameter (pixels). <=0 disables.
  static gtsam::SharedNoiseModel makeNoiseModel(
      const Eigen::Matrix2d& information, double cauchyParam = 0.0) {
    gtsam::SharedNoiseModel gaussian =
        gtsam::noiseModel::Gaussian::Information(information);
    if (cauchyParam > 0.0) {
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(cauchyParam), gaussian);
    }
    return gaussian;
  }

 private:
  Eigen::Vector2d measurement_;
  std::shared_ptr<const camera_geometry_t> cameraGeometry_;
};

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_GTSAMREPROJECTIONFACTOR_HPP_
