/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file GtsamConversions.hpp
 * @brief Conversions between OKVIS and GTSAM state representations, plus the
 *        variable key symbols used by the experimental GTSAM/DM-VIO backend.
 *
 * @warning The single most error-prone conversion here is the IMU bias: OKVIS
 *          stores SpeedAndBias as [v(3), b_g(3), b_a(3)] (gyro first), whereas
 *          gtsam::imuBias::ConstantBias is constructed as (b_a, b_g) (accel
 *          first). All bias conversions go through this file so the ordering
 *          flip lives in exactly one place. See TestGtsamImu.cpp.
 */

#ifndef INCLUDE_OKVIS_GTSAM_GTSAMCONVERSIONS_HPP_
#define INCLUDE_OKVIS_GTSAM_GTSAMCONVERSIONS_HPP_

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>

#include <okvis/FrameTypedefs.hpp>  // okvis::SpeedAndBias typedef
#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief gtsam_backend Namespace for the experimental GTSAM/DM-VIO backend.
namespace gtsam_backend {

// ---------------------------------------------------------------------------
// Variable key symbols. One symbol family per state quantity; the numeric part
// is the okvis StateId (or landmark/camera index).
//   X = pose T_WS, V = velocity v_W, B = bias, E = extrinsics T_SC, L = landmark
// ---------------------------------------------------------------------------
inline gtsam::Key poseKey(std::uint64_t i) { return gtsam::Symbol('x', i); }
inline gtsam::Key velocityKey(std::uint64_t i) { return gtsam::Symbol('v', i); }
inline gtsam::Key biasKey(std::uint64_t i) { return gtsam::Symbol('b', i); }
inline gtsam::Key extrinsicsKey(std::uint64_t i) { return gtsam::Symbol('e', i); }
inline gtsam::Key landmarkKey(std::uint64_t i) { return gtsam::Symbol('l', i); }

// ---------------------------------------------------------------------------
// Pose: okvis::kinematics::Transformation (T_WS, S->W) <-> gtsam::Pose3.
// Both store T_WS directly; no frame inversion needed at the boundary.
// ---------------------------------------------------------------------------
inline gtsam::Pose3 toPose3(const okvis::kinematics::Transformation& T_WS) {
  return gtsam::Pose3(gtsam::Rot3(Eigen::Quaterniond(T_WS.q())),
                      gtsam::Point3(T_WS.r()));
}
inline okvis::kinematics::Transformation fromPose3(const gtsam::Pose3& T_WS) {
  return okvis::kinematics::Transformation(T_WS.translation(),
                                           T_WS.rotation().toQuaternion());
}

// ---------------------------------------------------------------------------
// SpeedAndBias [v(3), b_g(3), b_a(3)] <-> (velocity, ConstantBias(b_a, b_g)).
// ---------------------------------------------------------------------------
inline Eigen::Vector3d velocityOf(const okvis::SpeedAndBias& sb) {
  return sb.head<3>();
}
/// \brief Extract the gtsam bias, applying the [gyro,accel] -> (accel,gyro) flip.
inline gtsam::imuBias::ConstantBias biasOf(const okvis::SpeedAndBias& sb) {
  const Eigen::Vector3d b_g = sb.segment<3>(3);
  const Eigen::Vector3d b_a = sb.segment<3>(6);
  return gtsam::imuBias::ConstantBias(b_a, b_g);
}
/// \brief Assemble an okvis SpeedAndBias from a world velocity and a gtsam bias.
inline okvis::SpeedAndBias toSpeedAndBias(const Eigen::Vector3d& v_W,
                                          const gtsam::imuBias::ConstantBias& b) {
  okvis::SpeedAndBias sb;
  sb.head<3>() = v_W;
  sb.segment<3>(3) = b.gyroscope();
  sb.tail<3>() = b.accelerometer();
  return sb;
}

// ---------------------------------------------------------------------------
// Landmarks: okvis homogeneous 4D point (W frame) <-> gtsam::Point3 (Euclidean).
// Valid for the stereo target where landmarks have finite metric depth.
// ---------------------------------------------------------------------------
inline gtsam::Point3 toPoint3(const Eigen::Vector4d& hp_W) {
  return gtsam::Point3(hp_W.head<3>() / hp_W[3]);
}
inline Eigen::Vector4d toHomogeneous(const gtsam::Point3& p_W) {
  Eigen::Vector4d hp_W;
  hp_W << p_W.x(), p_W.y(), p_W.z(), 1.0;
  return hp_W;
}

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_GTSAMCONVERSIONS_HPP_
