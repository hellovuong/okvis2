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

#include <algorithm>
#include <vector>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <okvis/gtsam/Marginalization.hpp>

namespace okvis {

namespace {
namespace gb = okvis::gtsam_backend;
}  // namespace

GtsamBackend::GtsamBackend(const okvis::ImuParameters& imuParameters)
    : imuParameters_(imuParameters),
      imuParams_(gb::makeCombinedParams(imuParameters)) {}

void GtsamBackend::addRawFactor(
    const gtsam::NonlinearFactor::shared_ptr& factor) {
  if (!factor) return;
  graph_.push_back(factor);
  delayed_.addFactor(factor);
}

int GtsamBackend::addImu(const okvis::ImuParameters& imuParameters) {
  imuParameters_ = imuParameters;
  imuParams_ = gb::makeCombinedParams(imuParameters);
  return 0;
}

int GtsamBackend::addCamera(const okvis::CameraParameters& cameraParameters) {
  cameraParams_.push_back(cameraParameters);
  return static_cast<int>(cameraParams_.size()) - 1;
}

bool GtsamBackend::addStates(okvis::MultiFramePtr multiFrame,
                             const okvis::ImuMeasurementDeque& imuMeasurements,
                             bool asKeyframe) {
  return addPropagatedState(StateId(multiFrame->id()), multiFrame->timestamp(),
                            imuMeasurements, asKeyframe);
}

bool GtsamBackend::addPropagatedState(StateId id, const okvis::Time& timestamp,
                                      const okvis::ImuMeasurementDeque& imu,
                                      bool asKeyframe) {
  const okvis::SpeedAndBias biasFromParams =
      gb::toSpeedAndBias(Eigen::Vector3d::Zero(),
                         gtsam::imuBias::ConstantBias(imuParameters_.a0,
                                                      imuParameters_.g0));

  if (states_.empty()) {
    // First frame: gravity-align orientation from the mean specific force.
    okvis::kinematics::Transformation T_WS;  // identity
    if (!imu.empty()) {
      Eigen::Vector3d acc = Eigen::Vector3d::Zero();
      for (const auto& m : imu) acc += m.measurement.accelerometers;
      acc /= static_cast<double>(imu.size());
      if (acc.norm() > 1e-6) {
        const Eigen::Quaterniond q_WS = Eigen::Quaterniond::FromTwoVectors(
            acc.normalized(), Eigen::Vector3d::UnitZ());
        T_WS = okvis::kinematics::Transformation(Eigen::Vector3d::Zero(), q_WS);
      }
    }
    addState(id, T_WS, biasFromParams, timestamp, asKeyframe);
    // Gauge priors: anchor position tightly, orientation loosely (IMU/visual fix it).
    addPosePrior(id, T_WS, 1e-3, 1e-1);
    addSpeedAndBiasPrior(id, biasFromParams, 0.1, imuParameters_.sigma_bg,
                         imuParameters_.sigma_ba);
    return true;
  }

  // Subsequent frame: IMU-propagate from the previous state for the initial guess.
  const StateId prev = currentStateId();
  const okvis::Time prevTs = stateMeta_.at(prev.value()).timestamp;
  const okvis::SpeedAndBias prevSb = getSpeedAndBias(prev);
  const gtsam::imuBias::ConstantBias prevBias = gb::biasOf(prevSb);
  const gtsam::NavState prevState(gb::toPose3(getPose(prev)),
                                  gb::velocityOf(prevSb));
  const gtsam::PreintegratedCombinedMeasurements pim =
      gb::preintegrate(imu, imuParameters_, prevBias, prevTs, timestamp);
  const gtsam::NavState predicted = pim.predict(prevState, prevBias);

  const okvis::kinematics::Transformation T_WS = gb::fromPose3(predicted.pose());
  const okvis::SpeedAndBias sb = gb::toSpeedAndBias(predicted.velocity(), prevBias);
  addState(id, T_WS, sb, timestamp, asKeyframe);
  addImuFactor(prev, id, imu, prevTs, timestamp);
  return true;
}

void GtsamBackend::setExtrinsics(std::size_t cameraId,
                                 const okvis::kinematics::Transformation& T_SC,
                                 bool fixed) {
  const gtsam::Key key = gb::extrinsicsKey(cameraId);
  const gtsam::Pose3 pose = gb::toPose3(T_SC);
  gtsam::Values v;
  v.insert(key, pose);
  if (extrinsics_.count(cameraId) == 0) {
    values_.insert(key, pose);
    delayed_.addValues(v);
    extrinsics_.insert(cameraId);
  } else {
    values_.update(key, pose);
    delayed_.updateValues(v);
  }
  // Anchor the extrinsics: very tight when fixed, looser when online-calibrated.
  const double sigma_t = fixed ? 1e-6 : 0.03;
  const double sigma_r = fixed ? 1e-6 : 0.3;
  gtsam::Vector6 sigmas;
  sigmas << sigma_r, sigma_r, sigma_r, sigma_t, sigma_t, sigma_t;  // [rot, trans]
  addRawFactor(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      key, pose, gtsam::noiseModel::Diagonal::Sigmas(sigmas)));
}

void GtsamBackend::addState(StateId id,
                            const okvis::kinematics::Transformation& T_WS,
                            const okvis::SpeedAndBias& speedAndBias,
                            const okvis::Time& timestamp, bool isKeyframe) {
  const std::uint64_t i = id.value();
  stateMeta_[i] = StateMeta{timestamp, isKeyframe};
  gtsam::Values v;
  v.insert(gb::poseKey(i), gb::toPose3(T_WS));
  v.insert(gb::velocityKey(i), gb::velocityOf(speedAndBias));
  v.insert(gb::biasKey(i), gb::biasOf(speedAndBias));
  if (states_.count(i)) {
    values_.update(gb::poseKey(i), gb::toPose3(T_WS));
    values_.update(gb::velocityKey(i), gb::velocityOf(speedAndBias));
    values_.update(gb::biasKey(i), gb::biasOf(speedAndBias));
    delayed_.updateValues(v);
    return;
  }
  values_.insert(gb::poseKey(i), gb::toPose3(T_WS));
  values_.insert(gb::velocityKey(i), gb::velocityOf(speedAndBias));
  values_.insert(gb::biasKey(i), gb::biasOf(speedAndBias));
  delayed_.addValues(v);
  states_.insert(i);
  imuFrames_.insert(id);
  if (isKeyframe) keyFrames_.insert(id);
}

okvis::Time GtsamBackend::timestamp(StateId id) const {
  const auto it = stateMeta_.find(id.value());
  return it == stateMeta_.end() ? okvis::Time(0) : it->second.timestamp;
}

bool GtsamBackend::isKeyframe(StateId id) const {
  const auto it = stateMeta_.find(id.value());
  return it != stateMeta_.end() && it->second.isKeyframe;
}

void GtsamBackend::setKeyframe(StateId id, bool isKeyframe) {
  const auto it = stateMeta_.find(id.value());
  if (it != stateMeta_.end()) it->second.isKeyframe = isKeyframe;
}

StateId GtsamBackend::currentStateId() const {
  return states_.empty() ? StateId() : StateId(*states_.rbegin());
}

StateId GtsamBackend::stateIdByAge(std::size_t age) const {
  if (age >= states_.size()) return StateId();
  auto it = states_.rbegin();
  std::advance(it, age);
  return StateId(*it);
}

void GtsamBackend::addPosePrior(StateId id,
                                const okvis::kinematics::Transformation& T_WS,
                                double sigma_translation,
                                double sigma_orientation) {
  gtsam::Vector6 sigmas;
  sigmas << sigma_orientation, sigma_orientation, sigma_orientation,
      sigma_translation, sigma_translation, sigma_translation;  // [rot, trans]
  addRawFactor(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gb::poseKey(id.value()), gb::toPose3(T_WS),
      gtsam::noiseModel::Diagonal::Sigmas(sigmas)));
}

void GtsamBackend::addSpeedAndBiasPrior(StateId id,
                                        const okvis::SpeedAndBias& speedAndBias,
                                        double sigma_v, double sigma_bg,
                                        double sigma_ba) {
  const std::uint64_t i = id.value();
  addRawFactor(boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
      gb::velocityKey(i), gb::velocityOf(speedAndBias),
      gtsam::noiseModel::Isotropic::Sigma(3, sigma_v)));
  gtsam::Vector6 biasSigmas;
  biasSigmas << sigma_ba, sigma_ba, sigma_ba, sigma_bg, sigma_bg, sigma_bg;  // [a, g]
  addRawFactor(boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
      gb::biasKey(i), gb::biasOf(speedAndBias),
      gtsam::noiseModel::Diagonal::Sigmas(biasSigmas)));
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
  addRawFactor(boost::make_shared<gtsam::CombinedImuFactor>(
      gb::poseKey(i), gb::velocityKey(i), gb::poseKey(j), gb::velocityKey(j),
      gb::biasKey(i), gb::biasKey(j), pim));
}

void GtsamBackend::addLandmark(LandmarkId id, const Eigen::Vector4d& hp_W) {
  const std::uint64_t i = id.value();
  const gtsam::Point3 p = gb::toPoint3(hp_W);
  gtsam::Values v;
  v.insert(gb::landmarkKey(i), p);
  if (landmarks_.count(i)) {
    values_.update(gb::landmarkKey(i), p);
    delayed_.updateValues(v);
    return;
  }
  values_.insert(gb::landmarkKey(i), p);
  delayed_.addValues(v);
  landmarks_.insert(i);
}

void GtsamBackend::marginalizeKeys(const gtsam::KeyVector& keysToDrop) {
  if (keysToDrop.empty()) return;
  const std::set<gtsam::Key> dropSet(keysToDrop.begin(), keysToDrop.end());

  // Partition factors: those touching any dropped key vs. the rest.
  gtsam::NonlinearFactorGraph touching, keep;
  for (const auto& factor : graph_) {
    if (!factor) continue;
    bool touches = false;
    for (const gtsam::Key k : factor->keys()) {
      if (dropSet.count(k)) {
        touches = true;
        break;
      }
    }
    if (touches) {
      touching.push_back(factor);
    } else {
      keep.push_back(factor);
    }
  }

  // Build the Schur-complement prior over the separator and rebuild the graph.
  const gtsam::NonlinearFactor::shared_ptr prior =
      gb::marginalizeOut(touching, values_, keysToDrop);
  graph_ = keep;
  if (prior) {
    graph_.push_back(prior);
    activePriors_.push_back(prior);
  }

  // Drop the marginalized variables from the ACTIVE estimate and bookkeeping.
  for (const gtsam::Key k : keysToDrop) {
    if (values_.exists(k)) values_.erase(k);
    const gtsam::Symbol sym(k);
    if (sym.chr() == 'x') {
      states_.erase(sym.index());
      stateMeta_.erase(sym.index());
      keyFrames_.erase(StateId(sym.index()));
      imuFrames_.erase(StateId(sym.index()));
    } else if (sym.chr() == 'l') {
      landmarks_.erase(sym.index());
    }
  }

  // Delayed marginalization: the delayed graph KEEPS the raw factors of the
  // dropped keys for `delayedLag_` keyframes so the prior can be re-derived
  // later (marginalization replacement). Beyond the lag, collapse them in the
  // delayed graph too. With lag 0 the delayed graph tracks the active one.
  if (delayedLag_ > 0) {
    droppedBatches_.push_back(keysToDrop);
    for (const gtsam::Key k : keysToDrop) retainedDroppedKeys_.push_back(k);
    while (static_cast<int>(droppedBatches_.size()) > delayedLag_) {
      const gtsam::KeyVector aged = droppedBatches_.front();
      droppedBatches_.pop_front();
      delayed_.advance(aged);
      const std::set<gtsam::Key> agedSet(aged.begin(), aged.end());
      gtsam::KeyVector remaining;
      for (const gtsam::Key k : retainedDroppedKeys_) {
        if (!agedSet.count(k)) remaining.push_back(k);
      }
      retainedDroppedKeys_.swap(remaining);
    }
  } else {
    delayed_.advance(keysToDrop);
  }
}

void GtsamBackend::enableDelayedMarginalization(int lag) {
  delayedLag_ = lag < 0 ? 0 : lag;
}

void GtsamBackend::remarginalize() {
  if (delayedLag_ <= 0 || retainedDroppedKeys_.empty() || remargDisabled_) return;

  // Refresh the delayed graph's live-key linearization with the current
  // (possibly bias/gravity-corrected) estimate, then re-derive the boundary
  // prior over the separator from the retained raw factors.
  delayed_.updateValues(values_);
  const gtsam::NonlinearFactor::shared_ptr newPrior =
      delayed_.recomputeBoundaryPrior(retainedDroppedKeys_, delayed_.values());
  if (!newPrior) return;

  // Swap: rebuild the active graph without the stale priors, add the fresh one.
  std::set<const gtsam::NonlinearFactor*> stale;
  for (const auto& p : activePriors_) stale.insert(p.get());
  gtsam::NonlinearFactorGraph rebuilt;
  for (const auto& f : graph_) {
    if (f && stale.count(f.get()) == 0) rebuilt.push_back(f);
  }
  rebuilt.push_back(newPrior);
  graph_ = rebuilt;
  activePriors_.clear();
  activePriors_.push_back(newPrior);
}

bool GtsamBackend::maybeRemarginalize(
    const gtsam::imuBias::ConstantBias& currentBias, double nowSec,
    double biasThreshold, double minIntervalSec) {
  if (delayedLag_ <= 0 || remargDisabled_) return false;
  const Eigen::Matrix<double, 6, 1> b = currentBias.vector();
  if (!haveRemargBias_) {
    biasAtLastRemarg_ = b;
    haveRemargBias_ = true;
    lastRemargTimeSec_ = nowSec;
    return false;
  }
  const double change = (b - biasAtLastRemarg_).norm();
  if (change < biasThreshold) return false;
  // Throttle; repeated throttle hits trip the circuit-breaker.
  if (nowSec - lastRemargTimeSec_ < minIntervalSec) {
    if (++remargThrottleHits_ >= 3) remargDisabled_ = true;
    return false;
  }
  remargThrottleHits_ = 0;
  remarginalize();
  biasAtLastRemarg_ = b;
  lastRemargTimeSec_ = nowSec;
  return true;
}

void GtsamBackend::rewriteAfterInit(
    const gtsam::Rot3& R_gw, const gtsam::imuBias::ConstantBias& bias,
    const std::map<std::uint64_t, Eigen::Vector3d>& velocities) {
  // Rotate every state and landmark from the visual world into the
  // gravity-aligned world (so world-z aligns with the IMU factor's gravity).
  auto rotateInto = [&](gtsam::Values& vals) {
    const gtsam::KeyVector keys = vals.keys();
    for (const gtsam::Key k : keys) {
      const gtsam::Symbol sym(k);
      switch (sym.chr()) {
        case 'x': {  // pose T_WS
          const gtsam::Pose3 T = vals.at<gtsam::Pose3>(k);
          vals.update(k, gtsam::Pose3(R_gw * T.rotation(),
                                      R_gw.rotate(T.translation())));
          break;
        }
        case 'v': {  // velocity (overwrite with recovered, rotated into G)
          const auto it = velocities.find(sym.index());
          const Eigen::Vector3d vW = (it != velocities.end())
                                         ? it->second
                                         : vals.at<Eigen::Vector3d>(k);
          vals.update(k, Eigen::Vector3d(R_gw.rotate(vW)));
          break;
        }
        case 'b':  // bias (set to recovered)
          vals.update(k, bias);
          break;
        case 'l': {  // landmark point
          const gtsam::Point3 p = vals.at<gtsam::Point3>(k);
          vals.update(k, gtsam::Point3(R_gw.rotate(p)));
          break;
        }
        default:
          break;  // extrinsics (camera-to-body) are unaffected by world rotation
      }
    }
  };
  rotateInto(values_);
  // Keep the delayed graph's linearization consistent for any later re-marg.
  gtsam::Values delayedVals = delayed_.values();
  rotateInto(delayedVals);
  delayed_.updateValues(delayedVals);
}

void GtsamBackend::marginalizeState(StateId id,
                                    const std::vector<LandmarkId>& alsoDrop) {
  gtsam::KeyVector keys;
  keys.push_back(gb::poseKey(id.value()));
  keys.push_back(gb::velocityKey(id.value()));
  keys.push_back(gb::biasKey(id.value()));
  for (const LandmarkId& l : alsoDrop) {
    keys.push_back(gb::landmarkKey(l.value()));
  }
  marginalizeKeys(keys);
}

bool GtsamBackend::applyStrategy(std::size_t numKeyframes,
                                 std::size_t /*numLoopClosureFrames*/,
                                 std::size_t numImuFrames,
                                 std::set<StateId>& affectedFrames, bool /*expand*/) {
  // Keep the newest numImuFrames states + the newest numKeyframes keyframes.
  std::set<std::uint64_t> keep;
  for (auto it = states_.rbegin(); it != states_.rend() && keep.size() < numImuFrames;
       ++it) {
    keep.insert(*it);
  }
  std::size_t kfKept = 0;
  for (auto it = states_.rbegin(); it != states_.rend() && kfKept < numKeyframes; ++it) {
    if (keyFrames_.count(StateId(*it))) {
      keep.insert(*it);
      ++kfKept;
    }
  }
  // Marginalize the rest, oldest first (states_ iterates ascending = oldest first).
  const std::vector<std::uint64_t> ordered(states_.begin(), states_.end());
  for (const std::uint64_t id : ordered) {
    if (keep.count(id) == 0) {
      marginalizeState(StateId(id));
      affectedFrames.insert(StateId(id));
    }
  }
  // Refresh the IMU window to the newest numImuFrames survivors.
  imuFrames_.clear();
  for (auto it = states_.rbegin();
       it != states_.rend() && imuFrames_.size() < numImuFrames; ++it) {
    imuFrames_.insert(StateId(*it));
  }
  return true;
}

void GtsamBackend::optimiseRealtimeGraph(int numIter,
                                         std::vector<StateId>& updatedStates,
                                         int /*numThreads*/, bool /*verbose*/,
                                         bool /*onlyNewestState*/,
                                         bool /*isInitialised*/) {
  optimise(std::max(numIter, optMinIterations_));
  updatedStates.clear();
  updatedStates.reserve(states_.size());
  for (const std::uint64_t id : states_) updatedStates.push_back(StateId(id));
}

bool GtsamBackend::setOptimisationTimeLimit(double timeLimit, int minIterations) {
  optTimeLimit_ = timeLimit;
  optMinIterations_ = minIterations;
  return true;
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
