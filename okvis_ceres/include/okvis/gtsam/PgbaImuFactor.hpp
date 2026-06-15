/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file PgbaImuFactor.hpp
 * @brief Gravity-variable IMU factor with FREE poses, for DM-VIO Pose-Graph
 *        Bundle Adjustment (PGBA) initialization.
 *
 * Unlike GravityAlignmentFactor (which holds the keyframe poses fixed), this
 * factor takes the two poses as optimisation variables so PGBA can move them
 * under the visual marginalization prior while jointly recovering gravity
 * direction (R_wg), velocities and bias. This is the ingredient that makes the
 * init true PGBA rather than ORB-SLAM3-style inertial-only alignment: the visual
 * uncertainty enters through a marginalization prior on these same pose
 * variables, and the IMU enters here.
 *
 * Residual = Forster preintegration error with g_W = R_wg * (0,0,-g):
 *   r_R = Log( dR(b)^T R_i^T R_j )
 *   r_p = R_i^T ( p_j - p_i - v_i dt - 1/2 g_W dt^2 ) - dP(b)
 *   r_v = R_i^T ( v_j - v_i - g_W dt )               - dV(b)
 * Jacobians are numerical (PGBA is a small, one-off solve).
 */

#ifndef INCLUDE_OKVIS_GTSAM_PGBAIMUFACTOR_HPP_
#define INCLUDE_OKVIS_GTSAM_PGBAIMUFACTOR_HPP_

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

/// \brief Gravity-variable IMU factor over (R_wg, T_i, v_i, T_j, v_j, bias).
class PgbaImuFactor
    : public gtsam::NoiseModelFactor6<gtsam::Rot3, gtsam::Pose3, gtsam::Vector3,
                                      gtsam::Pose3, gtsam::Vector3,
                                      gtsam::imuBias::ConstantBias> {
 public:
  typedef gtsam::NoiseModelFactor6<gtsam::Rot3, gtsam::Pose3, gtsam::Vector3,
                                   gtsam::Pose3, gtsam::Vector3,
                                   gtsam::imuBias::ConstantBias>
      Base;

  PgbaImuFactor(const gtsam::SharedNoiseModel& model, gtsam::Key gravityKey,
                gtsam::Key poseKeyI, gtsam::Key velKeyI, gtsam::Key poseKeyJ,
                gtsam::Key velKeyJ, gtsam::Key biasKey,
                const gtsam::PreintegratedCombinedMeasurements& pim,
                double gravity)
      : Base(model, gravityKey, poseKeyI, velKeyI, poseKeyJ, velKeyJ, biasKey),
        pim_(pim),
        gravity_(gravity) {}

  ~PgbaImuFactor() override = default;

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new PgbaImuFactor(*this));
  }

  static gtsam::SharedNoiseModel makeNoiseModel(
      const gtsam::PreintegratedCombinedMeasurements& pim) {
    const Eigen::Matrix<double, 9, 9> cov =
        pim.preintMeasCov().topLeftCorner<9, 9>();
    return gtsam::noiseModel::Gaussian::Covariance(cov);
  }

  gtsam::Vector evaluateError(
      const gtsam::Rot3& R_wg, const gtsam::Pose3& T_i, const gtsam::Vector3& v_i,
      const gtsam::Pose3& T_j, const gtsam::Vector3& v_j,
      const gtsam::imuBias::ConstantBias& bias,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none,
      boost::optional<gtsam::Matrix&> H3 = boost::none,
      boost::optional<gtsam::Matrix&> H4 = boost::none,
      boost::optional<gtsam::Matrix&> H5 = boost::none,
      boost::optional<gtsam::Matrix&> H6 = boost::none) const override {
    const gtsam::Vector9 r = residual(R_wg, T_i, v_i, T_j, v_j, bias);
    if (H1 || H2 || H3 || H4 || H5 || H6) {
      const std::function<gtsam::Vector9(
          const gtsam::Rot3&, const gtsam::Pose3&, const gtsam::Vector3&,
          const gtsam::Pose3&, const gtsam::Vector3&,
          const gtsam::imuBias::ConstantBias&)>
          f = [this](const gtsam::Rot3& a, const gtsam::Pose3& b,
                     const gtsam::Vector3& c, const gtsam::Pose3& d,
                     const gtsam::Vector3& e,
                     const gtsam::imuBias::ConstantBias& g) {
            return residual(a, b, c, d, e, g);
          };
      if (H1) *H1 = gtsam::numericalDerivative61<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
      if (H2) *H2 = gtsam::numericalDerivative62<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
      if (H3) *H3 = gtsam::numericalDerivative63<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
      if (H4) *H4 = gtsam::numericalDerivative64<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
      if (H5) *H5 = gtsam::numericalDerivative65<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
      if (H6) *H6 = gtsam::numericalDerivative66<gtsam::Vector9, gtsam::Rot3,
                        gtsam::Pose3, gtsam::Vector3, gtsam::Pose3,
                        gtsam::Vector3, gtsam::imuBias::ConstantBias>(
                        f, R_wg, T_i, v_i, T_j, v_j, bias);
    }
    return r;
  }

 private:
  gtsam::Vector9 residual(const gtsam::Rot3& R_wg, const gtsam::Pose3& T_i,
                          const gtsam::Vector3& v_i, const gtsam::Pose3& T_j,
                          const gtsam::Vector3& v_j,
                          const gtsam::imuBias::ConstantBias& bias) const {
    const double dt = pim_.deltaTij();
    const gtsam::Vector3 gW = R_wg.rotate(gtsam::Vector3(0.0, 0.0, -gravity_));

    const gtsam::Vector9 xi = pim_.biasCorrectedDelta(bias);
    const gtsam::Rot3 dR = gtsam::Rot3::Expmap(xi.head<3>());
    const gtsam::Vector3 dP = xi.segment<3>(3);
    const gtsam::Vector3 dV = xi.tail<3>();

    const gtsam::Rot3& Ri = T_i.rotation();
    const gtsam::Rot3& Rj = T_j.rotation();
    const gtsam::Vector3 pi = T_i.translation();
    const gtsam::Vector3 pj = T_j.translation();
    const Eigen::Matrix3d RiT = Ri.transpose();

    gtsam::Vector9 r;
    r.head<3>() = gtsam::Rot3::Logmap(dR.inverse() * (Ri.between(Rj)));
    r.segment<3>(3) = RiT * (pj - pi - v_i * dt - 0.5 * gW * dt * dt) - dP;
    r.tail<3>() = RiT * (v_j - v_i - gW * dt) - dV;
    return r;
  }

  gtsam::PreintegratedCombinedMeasurements pim_;
  double gravity_;
};

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_PGBAIMUFACTOR_HPP_
