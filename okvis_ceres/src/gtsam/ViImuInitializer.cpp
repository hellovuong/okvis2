/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file ViImuInitializer.cpp
 * @brief Implementation of the DM-VIO-style dynamic IMU initializer.
 */

#include <okvis/ViImuInitializer.hpp>

#include <boost/make_shared.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <okvis/gtsam/GravityAlignmentFactor.hpp>
#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>

namespace okvis {

namespace {
namespace gb = okvis::gtsam_backend;

gtsam::Key gravityKey() { return gtsam::Symbol('a', 0); }
gtsam::Key velKey(std::uint64_t i) { return gtsam::Symbol('v', i); }
gtsam::Key biasKey() { return gtsam::Symbol('b', 0); }

// Rotation aligning unit vector `from` to unit vector `to`.
gtsam::Rot3 alignVectors(const Eigen::Vector3d& from, const Eigen::Vector3d& to) {
  const Eigen::Vector3d a = from.normalized();
  const Eigen::Vector3d b = to.normalized();
  const double c = a.dot(b);
  if (c > 1.0 - 1e-9) return gtsam::Rot3();
  if (c < -1.0 + 1e-9) {
    // 180 deg: rotate about any axis perpendicular to a.
    Eigen::Vector3d axis = a.unitOrthogonal();
    return gtsam::Rot3::AxisAngle(axis, M_PI);
  }
  const Eigen::Vector3d axis = a.cross(b).normalized();
  const double angle = std::acos(std::max(-1.0, std::min(1.0, c)));
  return gtsam::Rot3::AxisAngle(axis, angle);
}

}  // namespace

ViImuInitializer::ViImuInitializer(const okvis::ImuParameters& imuParameters)
    : imuParameters_(imuParameters) {
  state_ = (imuParameters_.initStrategy == ImuParameters::InitStrategy::Dynamic)
               ? State::VisualOnly
               : State::Static;
}

void ViImuInitializer::addKeyframe(
    StateId id, const okvis::kinematics::Transformation& T_WS,
    const okvis::ImuMeasurementDeque& imuSincePrev, const okvis::Time& tPrev,
    const okvis::Time& tCurr) {
  if (!started_) {
    started_ = true;
    startTime_ = tCurr;
  }

  Frame f;
  f.id = id;
  f.T_WS = T_WS;
  f.timestamp = tCurr;
  if (!window_.empty() && !imuSincePrev.empty() && tCurr > tPrev) {
    const gtsam::imuBias::ConstantBias bias(imuParameters_.a0, imuParameters_.g0);
    f.pim = boost::make_shared<gtsam::PreintegratedCombinedMeasurements>(
        gb::preintegrate(imuSincePrev, imuParameters_, bias, tPrev, tCurr));
  }
  window_.push_back(f);

  for (const auto& m : imuSincePrev) {
    gyrSamples_.push_back(m.measurement.gyroscopes);
    accSamples_.push_back(m.measurement.accelerometers);
  }

  // Bound the window and the excitation buffers.
  const int maxFrames = std::max(3, imuParameters_.jointInitWindow);
  while (static_cast<int>(window_.size()) > maxFrames) window_.pop_front();
  const std::size_t maxSamples = 4000;
  while (gyrSamples_.size() > maxSamples) gyrSamples_.pop_front();
  while (accSamples_.size() > maxSamples) accSamples_.pop_front();
}

bool ViImuInitializer::excitationSufficient() const {
  if (gyrSamples_.size() < 10 || accSamples_.size() < 10) return false;

  auto stdev = [](const std::deque<Eigen::Vector3d>& s) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& v : s) mean += v;
    mean /= static_cast<double>(s.size());
    double var = 0.0;
    for (const auto& v : s) var += (v - mean).squaredNorm();
    return std::sqrt(var / static_cast<double>(s.size()));
  };

  const double gyrStd = stdev(gyrSamples_);
  const double accStd = stdev(accSamples_);

  // Visual translation across the window.
  double trans = 0.0;
  if (window_.size() >= 2) {
    trans = (window_.back().T_WS.r() - window_.front().T_WS.r()).norm();
  }

  return gyrStd > imuParameters_.excitationThreshGyr &&
         accStd > imuParameters_.excitationThreshAcc && trans > 0.10;
}

gtsam::Rot3 ViImuInitializer::initialGravityAlignment() const {
  // Mean specific force (body), lifted to the visual world by the first pose.
  Eigen::Vector3d fMean = Eigen::Vector3d::Zero();
  for (const auto& a : accSamples_) fMean += a;
  if (!accSamples_.empty()) fMean /= static_cast<double>(accSamples_.size());

  // For near-constant velocity, f ~ -R_WS^T g_W  =>  g_W ~ -R_WS fMean.
  const Eigen::Matrix3d R_WS0 = window_.front().T_WS.C();
  const Eigen::Vector3d gW = -(R_WS0 * fMean);
  // R_wg maps the aligned-down gravity (0,0,-g) to the estimated g_W direction.
  return alignVectors(Eigen::Vector3d(0, 0, -1), gW);
}

ViImuInitializer::Result ViImuInitializer::staticFallback() const {
  Result r;
  r.converged = true;
  r.usedStaticFallback = true;
  r.bias = gtsam::imuBias::ConstantBias(imuParameters_.a0, imuParameters_.g0);
  r.R_wg = window_.empty() ? gtsam::Rot3() : initialGravityAlignment();
  for (const auto& f : window_) r.velocities[f.id.value()] = Eigen::Vector3d::Zero();
  return r;
}

bool ViImuInitializer::runJointInit(Result* result) const {
  if (window_.size() < 3) return false;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  const gtsam::Rot3 R_wg0 = initialGravityAlignment();
  values.insert(gravityKey(), R_wg0);
  // Weak gauge prior on the gravity alignment (yaw about gravity unobservable).
  graph.addPrior(gravityKey(), R_wg0, gtsam::noiseModel::Isotropic::Sigma(3, 1.0));

  const gtsam::imuBias::ConstantBias bias0(imuParameters_.a0, imuParameters_.g0);
  values.insert(biasKey(), bias0);
  gtsam::Vector6 biasSigmas;
  biasSigmas << imuParameters_.sigma_ba, imuParameters_.sigma_ba,
      imuParameters_.sigma_ba, imuParameters_.sigma_bg, imuParameters_.sigma_bg,
      imuParameters_.sigma_bg;
  graph.addPrior(biasKey(), bias0, gtsam::noiseModel::Diagonal::Sigmas(biasSigmas));

  // Finite-difference initial velocities; insert one per keyframe.
  for (std::size_t i = 0; i < window_.size(); ++i) {
    Eigen::Vector3d v;
    if (i + 1 < window_.size()) {
      const double dt =
          (window_[i + 1].timestamp - window_[i].timestamp).toSec();
      v = (dt > 1e-6)
              ? Eigen::Vector3d((window_[i + 1].T_WS.r() - window_[i].T_WS.r()) / dt)
              : Eigen::Vector3d::Zero();
    } else if (window_.size() >= 2) {
      const double dt =
          (window_[i].timestamp - window_[i - 1].timestamp).toSec();
      v = (dt > 1e-6)
              ? Eigen::Vector3d((window_[i].T_WS.r() - window_[i - 1].T_WS.r()) / dt)
              : Eigen::Vector3d::Zero();
    } else {
      v = Eigen::Vector3d::Zero();
    }
    values.insert(velKey(window_[i].id.value()), v);
  }

  // Gravity-alignment factors between consecutive keyframes.
  for (std::size_t i = 1; i < window_.size(); ++i) {
    if (!window_[i].pim) continue;
    const auto& pim = *window_[i].pim;
    graph.emplace_shared<gtsam_backend::GravityAlignmentFactor>(
        gtsam_backend::GravityAlignmentFactor::makeNoiseModel(pim), gravityKey(),
        velKey(window_[i - 1].id.value()), velKey(window_[i].id.value()),
        biasKey(), gb::toPose3(window_[i - 1].T_WS), gb::toPose3(window_[i].T_WS),
        pim, imuParameters_.g);
  }

  gtsam::LevenbergMarquardtParams params;
  params.setMaxIterations(50);
  gtsam::Values solution;
  try {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    solution = optimizer.optimize();
  } catch (const std::exception&) {
    return false;
  }

  // Gate on gravity observability: smallest eigenvalue of the gravity info.
  try {
    gtsam::Marginals marginals(graph, solution);
    const Eigen::Matrix3d covG = marginals.marginalCovariance(gravityKey());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covG);
    const double largestCov = es.eigenvalues().maxCoeff();
    const double smallestInfo = (largestCov > 0.0) ? 1.0 / largestCov : 0.0;
    result->conditionNumber = smallestInfo;
    if (smallestInfo < imuParameters_.initMinCondition) return false;
  } catch (const std::exception&) {
    return false;
  }

  result->converged = true;
  result->usedStaticFallback = false;
  result->R_wg = solution.at<gtsam::Rot3>(gravityKey());
  result->bias = solution.at<gtsam::imuBias::ConstantBias>(biasKey());
  for (const auto& f : window_) {
    result->velocities[f.id.value()] =
        solution.at<Eigen::Vector3d>(velKey(f.id.value()));
  }
  return true;
}

ViImuInitializer::Result ViImuInitializer::step() {
  Result r;
  if (!started_) return r;

  if (imuParameters_.initStrategy == ImuParameters::InitStrategy::Static) {
    r = staticFallback();
    state_ = State::Converged;
    return r;
  }

  if (state_ == State::Converged) {
    r.converged = true;
    return r;
  }

  // Timeout fallback: no excitation after the configured window.
  const double elapsed = (window_.back().timestamp - startTime_).toSec();
  if (elapsed > imuParameters_.initTimeoutSec && !excitationSufficient()) {
    r = staticFallback();
    state_ = State::Converged;
    return r;
  }

  if (window_.size() >= 3 && excitationSufficient()) {
    state_ = State::JointInit;
    if (runJointInit(&r)) {
      state_ = State::Converged;
      return r;
    }
    // Gating failed: stay in VisualOnly and accumulate more excitation.
    state_ = State::VisualOnly;
  }
  return r;
}

}  // namespace okvis
