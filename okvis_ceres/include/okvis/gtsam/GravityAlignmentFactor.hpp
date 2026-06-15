/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file GravityAlignmentFactor.hpp
 * @brief IMU factor with the gravity direction as an optimizable variable, for
 *        DM-VIO-style dynamic IMU initialization (Phase B).
 *
 * Connects a world-frame gravity-alignment rotation R_wg (gtsam::Rot3) and the
 * velocities + (shared) bias of two keyframes, with the keyframe POSES held
 * fixed (stereo gives them metrically). The residual is the standard
 * Forster-style preintegration error evaluated with gravity g_W = R_wg * (0,0,-g):
 *
 *   r_R = Log( dR(b)^T  R_i^T R_j )
 *   r_p = R_i^T ( p_j - p_i - v_i dt - 1/2 g_W dt^2 ) - dP(b)
 *   r_v = R_i^T ( v_j - v_i - g_W dt )               - dV(b)
 *
 * where dR/dP/dV(b) are the bias-corrected preintegrated deltas (GTSAM's
 * biasCorrectedDelta). R_wg's yaw about gravity is unobservable; pin it with a
 * prior (the initializer does this). Jacobians are computed numerically — the
 * init solve is small and runs rarely, so this is an acceptable, robust choice.
 */

#ifndef INCLUDE_OKVIS_GTSAM_GRAVITYALIGNMENTFACTOR_HPP_
#define INCLUDE_OKVIS_GTSAM_GRAVITYALIGNMENTFACTOR_HPP_

#include <functional>

#include <Eigen/Core>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace okvis {
namespace gtsam_backend {

/// \brief Preintegration factor with gravity direction as a variable.
class GravityAlignmentFactor
    : public gtsam::NoiseModelFactor4<gtsam::Rot3, gtsam::Vector3, gtsam::Vector3,
                                      gtsam::imuBias::ConstantBias> {
 public:
  typedef gtsam::NoiseModelFactor4<gtsam::Rot3, gtsam::Vector3, gtsam::Vector3,
                                   gtsam::imuBias::ConstantBias>
      Base;

  /// \brief Construct.
  /// \param model      9D noise model (e.g. from makeNoiseModel(pim)).
  /// \param gravityKey Key of the R_wg gravity-alignment rotation.
  /// \param velKeyI    Key of v_i (world frame).
  /// \param velKeyJ    Key of v_j (world frame).
  /// \param biasKey    Key of the shared IMU bias.
  /// \param T_WS_i     Fixed pose of keyframe i.
  /// \param T_WS_j     Fixed pose of keyframe j.
  /// \param pim        Preintegrated combined measurements between i and j.
  /// \param gravity    Gravity magnitude [m/s^2].
  GravityAlignmentFactor(const gtsam::SharedNoiseModel& model,
                         gtsam::Key gravityKey, gtsam::Key velKeyI,
                         gtsam::Key velKeyJ, gtsam::Key biasKey,
                         const gtsam::Pose3& T_WS_i, const gtsam::Pose3& T_WS_j,
                         const gtsam::PreintegratedCombinedMeasurements& pim,
                         double gravity)
      : Base(model, gravityKey, velKeyI, velKeyJ, biasKey),
        T_i_(T_WS_i),
        T_j_(T_WS_j),
        pim_(pim),
        gravity_(gravity) {}

  ~GravityAlignmentFactor() override = default;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new GravityAlignmentFactor(*this));
  }

  /// \brief Build a 9D Gaussian noise model from the preintegration covariance.
  static gtsam::SharedNoiseModel makeNoiseModel(
      const gtsam::PreintegratedCombinedMeasurements& pim) {
    const Eigen::Matrix<double, 9, 9> cov =
        pim.preintMeasCov().topLeftCorner<9, 9>();
    return gtsam::noiseModel::Gaussian::Covariance(cov);
  }

  gtsam::Vector evaluateError(
      const gtsam::Rot3& R_wg, const gtsam::Vector3& v_i,
      const gtsam::Vector3& v_j, const gtsam::imuBias::ConstantBias& bias,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none,
      boost::optional<gtsam::Matrix&> H3 = boost::none,
      boost::optional<gtsam::Matrix&> H4 = boost::none) const override {
    const gtsam::Vector9 r = residual(R_wg, v_i, v_j, bias);
    if (H1 || H2 || H3 || H4) {
      const std::function<gtsam::Vector9(
          const gtsam::Rot3&, const gtsam::Vector3&, const gtsam::Vector3&,
          const gtsam::imuBias::ConstantBias&)>
          f = [this](const gtsam::Rot3& a, const gtsam::Vector3& b,
                     const gtsam::Vector3& c,
                     const gtsam::imuBias::ConstantBias& d) {
            return residual(a, b, c, d);
          };
      if (H1)
        *H1 = gtsam::numericalDerivative41<gtsam::Vector9, gtsam::Rot3,
                                           gtsam::Vector3, gtsam::Vector3,
                                           gtsam::imuBias::ConstantBias>(
            f, R_wg, v_i, v_j, bias);
      if (H2)
        *H2 = gtsam::numericalDerivative42<gtsam::Vector9, gtsam::Rot3,
                                           gtsam::Vector3, gtsam::Vector3,
                                           gtsam::imuBias::ConstantBias>(
            f, R_wg, v_i, v_j, bias);
      if (H3)
        *H3 = gtsam::numericalDerivative43<gtsam::Vector9, gtsam::Rot3,
                                           gtsam::Vector3, gtsam::Vector3,
                                           gtsam::imuBias::ConstantBias>(
            f, R_wg, v_i, v_j, bias);
      if (H4)
        *H4 = gtsam::numericalDerivative44<gtsam::Vector9, gtsam::Rot3,
                                           gtsam::Vector3, gtsam::Vector3,
                                           gtsam::imuBias::ConstantBias>(
            f, R_wg, v_i, v_j, bias);
    }
    return r;
  }

 private:
  gtsam::Vector9 residual(const gtsam::Rot3& R_wg, const gtsam::Vector3& v_i,
                          const gtsam::Vector3& v_j,
                          const gtsam::imuBias::ConstantBias& bias) const {
    const double dt = pim_.deltaTij();
    const gtsam::Vector3 gW = R_wg.rotate(gtsam::Vector3(0.0, 0.0, -gravity_));

    // Bias-corrected preintegrated delta: [theta(3), pos(3), vel(3)], frame i.
    const gtsam::Vector9 xi = pim_.biasCorrectedDelta(bias);
    const gtsam::Rot3 dR = gtsam::Rot3::Expmap(xi.head<3>());
    const gtsam::Vector3 dP = xi.segment<3>(3);
    const gtsam::Vector3 dV = xi.tail<3>();

    const gtsam::Rot3& Ri = T_i_.rotation();
    const gtsam::Rot3& Rj = T_j_.rotation();
    const gtsam::Vector3 pi = T_i_.translation();
    const gtsam::Vector3 pj = T_j_.translation();
    const Eigen::Matrix3d RiT = Ri.transpose();

    gtsam::Vector9 r;
    r.head<3>() = gtsam::Rot3::Logmap(dR.inverse() * (Ri.between(Rj)));
    r.segment<3>(3) = RiT * (pj - pi - v_i * dt - 0.5 * gW * dt * dt) - dP;
    r.tail<3>() = RiT * (v_j - v_i - gW * dt) - dV;
    return r;
  }

  gtsam::Pose3 T_i_;
  gtsam::Pose3 T_j_;
  gtsam::PreintegratedCombinedMeasurements pim_;
  double gravity_;
};

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_GRAVITYALIGNMENTFACTOR_HPP_
