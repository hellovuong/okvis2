/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file GtsamBackend.cpp
 * @brief Implementation of the experimental GTSAM visual-inertial backend.
 */

#include <okvis/GtsamBackend.hpp>

#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

namespace okvis {

namespace {
namespace gb = okvis::gtsam_backend;
}  // namespace

GtsamBackend::GtsamBackend(const okvis::ImuParameters& imuParameters)
    : imuParameters_(imuParameters),
      imuParams_(gb::makeCombinedParams(imuParameters)) {}

void GtsamBackend::setExtrinsics(std::size_t cameraId,
                                 const okvis::kinematics::Transformation& T_SC,
                                 bool fixed) {
  const gtsam::Key key = gb::extrinsicsKey(cameraId);
  const gtsam::Pose3 pose = gb::toPose3(T_SC);
  if (extrinsics_.count(cameraId) == 0) {
    values_.insert(key, pose);
    extrinsics_.insert(cameraId);
  } else {
    values_.update(key, pose);
  }
  // Anchor the extrinsics: very tight when fixed, looser when online-calibrated.
  const double sigma_t = fixed ? 1e-6 : 0.03;
  const double sigma_r = fixed ? 1e-6 : 0.3;
  gtsam::Vector6 sigmas;
  sigmas << sigma_r, sigma_r, sigma_r, sigma_t, sigma_t, sigma_t;  // [rot, trans]
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      key, pose, gtsam::noiseModel::Diagonal::Sigmas(sigmas));
}

void GtsamBackend::addState(StateId id,
                            const okvis::kinematics::Transformation& T_WS,
                            const okvis::SpeedAndBias& speedAndBias) {
  const std::uint64_t i = id.value();
  if (states_.count(i)) {
    values_.update(gb::poseKey(i), gb::toPose3(T_WS));
    values_.update(gb::velocityKey(i), gb::velocityOf(speedAndBias));
    values_.update(gb::biasKey(i), gb::biasOf(speedAndBias));
    return;
  }
  values_.insert(gb::poseKey(i), gb::toPose3(T_WS));
  values_.insert(gb::velocityKey(i), gb::velocityOf(speedAndBias));
  values_.insert(gb::biasKey(i), gb::biasOf(speedAndBias));
  states_.insert(i);
}

void GtsamBackend::addPosePrior(StateId id,
                                const okvis::kinematics::Transformation& T_WS,
                                double sigma_translation,
                                double sigma_orientation) {
  gtsam::Vector6 sigmas;
  sigmas << sigma_orientation, sigma_orientation, sigma_orientation,
      sigma_translation, sigma_translation, sigma_translation;  // [rot, trans]
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gb::poseKey(id.value()), gb::toPose3(T_WS),
      gtsam::noiseModel::Diagonal::Sigmas(sigmas));
}

void GtsamBackend::addSpeedAndBiasPrior(StateId id,
                                        const okvis::SpeedAndBias& speedAndBias,
                                        double sigma_v, double sigma_bg,
                                        double sigma_ba) {
  const std::uint64_t i = id.value();
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
      gb::velocityKey(i), gb::velocityOf(speedAndBias),
      gtsam::noiseModel::Isotropic::Sigma(3, sigma_v));
  gtsam::Vector6 biasSigmas;
  biasSigmas << sigma_ba, sigma_ba, sigma_ba, sigma_bg, sigma_bg, sigma_bg;  // [a, g]
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
      gb::biasKey(i), gb::biasOf(speedAndBias),
      gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));
}

void GtsamBackend::addImuFactor(StateId from, StateId to,
                                const okvis::ImuMeasurementDeque& imuMeasurements,
                                const okvis::Time& t0, const okvis::Time& t1) {
  const std::uint64_t i = from.value();
  const std::uint64_t j = to.value();
  // Linearization bias: current estimate at the "from" state.
  const gtsam::imuBias::ConstantBias bias =
      values_.at<gtsam::imuBias::ConstantBias>(gb::biasKey(i));
  const gtsam::PreintegratedCombinedMeasurements pim =
      gb::preintegrate(imuMeasurements, imuParameters_, bias, t0, t1);
  graph_.emplace_shared<gtsam::CombinedImuFactor>(
      gb::poseKey(i), gb::velocityKey(i), gb::poseKey(j), gb::velocityKey(j),
      gb::biasKey(i), gb::biasKey(j), pim);
}

void GtsamBackend::addLandmark(LandmarkId id, const Eigen::Vector4d& hp_W) {
  const std::uint64_t i = id.value();
  const gtsam::Point3 p = gb::toPoint3(hp_W);
  if (landmarks_.count(i)) {
    values_.update(gb::landmarkKey(i), p);
    return;
  }
  values_.insert(gb::landmarkKey(i), p);
  landmarks_.insert(i);
}

double GtsamBackend::optimise(int maxIterations) {
  gtsam::LevenbergMarquardtParams params;
  params.setMaxIterations(maxIterations);
  params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, params);
  values_ = optimizer.optimize();
  return graph_.error(values_);
}

okvis::kinematics::Transformation GtsamBackend::getPose(StateId id) const {
  return gb::fromPose3(values_.at<gtsam::Pose3>(gb::poseKey(id.value())));
}

okvis::SpeedAndBias GtsamBackend::getSpeedAndBias(StateId id) const {
  const std::uint64_t i = id.value();
  const gtsam::Vector3 v = values_.at<gtsam::Vector3>(gb::velocityKey(i));
  const gtsam::imuBias::ConstantBias b =
      values_.at<gtsam::imuBias::ConstantBias>(gb::biasKey(i));
  return gb::toSpeedAndBias(v, b);
}

Eigen::Vector4d GtsamBackend::getLandmark(LandmarkId id) const {
  return gb::toHomogeneous(values_.at<gtsam::Point3>(gb::landmarkKey(id.value())));
}

okvis::kinematics::Transformation GtsamBackend::getExtrinsics(
    std::size_t cameraId) const {
  return gb::fromPose3(values_.at<gtsam::Pose3>(gb::extrinsicsKey(cameraId)));
}

}  // namespace okvis
