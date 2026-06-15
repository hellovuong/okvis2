/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file GtsamBackend.hpp
 * @brief Experimental GTSAM factor-graph visual-inertial backend (Phase 3).
 *
 * Self-contained sliding-window backend: assembles a gtsam::NonlinearFactorGraph
 * over poses (X), velocities (V), IMU biases (B), camera extrinsics (E) and
 * landmarks (L), and solves it with batch Levenberg-Marquardt. This is the
 * foundation onto which DM-VIO marginalization (Phase 4) and the delayed-graph /
 * PGBA initializer (Phases A-C) are built. It does not yet replace ViSlamBackend
 * in ThreadedSlam; that wiring follows once this is validated on real sequences.
 *
 * States use okvis conventions throughout the public API (Transformation T_WS,
 * SpeedAndBias [v, b_g, b_a]); all conversions go through GtsamConversions.hpp.
 */

#ifndef INCLUDE_OKVIS_GTSAMBACKEND_HPP_
#define INCLUDE_OKVIS_GTSAMBACKEND_HPP_

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <boost/make_shared.hpp>

#include <Eigen/Core>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <deque>

#include <gtsam/geometry/Rot3.h>

#include <okvis/gtsam/DelayedGraph.hpp>
#include <okvis/gtsam/GtsamConversions.hpp>
#include <okvis/gtsam/GtsamReprojectionFactor.hpp>
#include <okvis/gtsam/ImuPreintegrationGtsam.hpp>

namespace okvis {

/// \brief A GTSAM factor-graph visual-inertial backend.
class GtsamBackend {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Construct with IMU parameters (for preintegration noise / gravity).
  explicit GtsamBackend(const okvis::ImuParameters& imuParameters);

  // --- extrinsics -----------------------------------------------------------
  /// \brief Add/define camera extrinsics T_SC. If fixed, anchors with a tight
  ///        prior (online calibration releases this later).
  void setExtrinsics(std::size_t cameraId,
                     const okvis::kinematics::Transformation& T_SC,
                     bool fixed = true);

  // --- states ---------------------------------------------------------------
  /// \brief Insert a state (pose + velocity + bias) as initial values.
  void addState(StateId id, const okvis::kinematics::Transformation& T_WS,
                const okvis::SpeedAndBias& speedAndBias);

  /// \brief Add a Gaussian prior on a pose (e.g. for gauge fixing the first KF).
  void addPosePrior(StateId id, const okvis::kinematics::Transformation& T_WS,
                    double sigma_translation, double sigma_orientation);

  /// \brief Add a Gaussian prior on a speed-and-bias state.
  void addSpeedAndBiasPrior(StateId id, const okvis::SpeedAndBias& speedAndBias,
                            double sigma_v, double sigma_bg, double sigma_ba);

  // --- IMU ------------------------------------------------------------------
  /// \brief Add a combined IMU factor between consecutive states, preintegrating
  ///        with the current bias estimate at the "from" state.
  void addImuFactor(StateId from, StateId to,
                    const okvis::ImuMeasurementDeque& imuMeasurements,
                    const okvis::Time& t0, const okvis::Time& t1);

  // --- landmarks ------------------------------------------------------------
  /// \brief Insert a landmark (homogeneous world point) as an initial value.
  void addLandmark(LandmarkId id, const Eigen::Vector4d& hp_W);

  /// \brief Add a stereo/mono reprojection observation.
  /// \tparam GEOMETRY_TYPE The OKVIS camera geometry type.
  template <class GEOMETRY_TYPE>
  void addObservation(StateId stateId, LandmarkId landmarkId,
                      std::size_t cameraId, const Eigen::Vector2d& keypoint,
                      const Eigen::Matrix2d& information,
                      std::shared_ptr<const GEOMETRY_TYPE> cameraGeometry,
                      double cauchyParam = 0.0) {
    typedef okvis::gtsam_backend::GtsamReprojectionFactor<GEOMETRY_TYPE> Factor;
    addRawFactor(boost::make_shared<Factor>(
        Factor::makeNoiseModel(information, cauchyParam),
        okvis::gtsam_backend::poseKey(stateId.value()),
        okvis::gtsam_backend::landmarkKey(landmarkId.value()),
        okvis::gtsam_backend::extrinsicsKey(cameraId), keypoint,
        std::move(cameraGeometry)));
  }

  // --- marginalisation ------------------------------------------------------
  /// \brief Marginalize a set of variables out of the sliding window.
  ///
  /// Partitions the graph into factors touching the dropped keys and the rest,
  /// replaces the former with a single Schur-complement prior
  /// (LinearContainerFactor) over the separator, and removes the dropped
  /// variables from the estimate. This is the GTSAM-native analogue of OKVIS's
  /// MST pose-graph conversion, but the resulting prior is relinearizable.
  /// \param keysToDrop GTSAM keys (use poseKey/velocityKey/biasKey/landmarkKey).
  void marginalizeKeys(const gtsam::KeyVector& keysToDrop);

  /// \brief Convenience: marginalize a full state (pose + velocity + bias).
  void marginalizeState(StateId id, const std::vector<LandmarkId>& alsoDrop = {});

  // --- DM-VIO delayed marginalization (Phase D) -----------------------------
  /// \brief Enable delayed marginalization with a lag of `lag` keyframes. The
  ///        delayed graph retains the raw factors of marginalized states within
  ///        the lag, so the active marginalization prior can be re-derived
  ///        (relinearized) after a bias/gravity correction. 0 disables (legacy).
  void enableDelayedMarginalization(int lag);

  /// \brief Re-derive the active marginalization prior from the retained raw
  ///        factors at the CURRENT linearization, and swap it into the active
  ///        graph (DM-VIO marginalization replacement). No-op without retained
  ///        drops or when delayed marginalization is disabled.
  void remarginalize();

  /// \brief Trigger remarginalize() when the aggregate bias change since the
  ///        last re-marginalization exceeds a threshold, subject to a throttle
  ///        and a circuit-breaker (disables after repeated throttle hits).
  /// \return True if a re-marginalization was performed.
  bool maybeRemarginalize(const gtsam::imuBias::ConstantBias& currentBias,
                          double nowSec, double biasThreshold,
                          double minIntervalSec);

  // --- post-init state rewrite (Phase C bridge) -----------------------------
  /// \brief Apply the result of dynamic IMU initialization: rotate all states
  ///        and landmarks from the (arbitrary) visual world into the
  ///        gravity-aligned world by R_gw (= R_wg^{-1}), set the IMU bias, and
  ///        overwrite the recovered per-keyframe velocities. After this the
  ///        world z-axis is gravity-aligned, matching the IMU factor params.
  /// \param R_gw        Rotation from visual world to gravity-aligned world.
  /// \param bias        Recovered IMU bias.
  /// \param velocities  Recovered per-StateId velocities (in the visual world).
  void rewriteAfterInit(const gtsam::Rot3& R_gw,
                        const gtsam::imuBias::ConstantBias& bias,
                        const std::map<std::uint64_t, Eigen::Vector3d>& velocities);

  // --- optimisation ---------------------------------------------------------
  /// \brief Run batch Levenberg-Marquardt and adopt the result as the estimate.
  /// \param maxIterations Maximum LM iterations.
  /// \return The total (whitened) error after optimisation.
  double optimise(int maxIterations = 10);

  /// \brief Total graph error at the current estimate.
  double error() const { return graph_.error(values_); }

  // --- getters --------------------------------------------------------------
  okvis::kinematics::Transformation getPose(StateId id) const;
  okvis::SpeedAndBias getSpeedAndBias(StateId id) const;
  Eigen::Vector4d getLandmark(LandmarkId id) const;
  okvis::kinematics::Transformation getExtrinsics(std::size_t cameraId) const;

  bool hasState(StateId id) const { return states_.count(id.value()) > 0; }
  bool hasLandmark(LandmarkId id) const { return landmarks_.count(id.value()) > 0; }

  /// \brief Access the underlying graph (for marginalization in later phases).
  const gtsam::NonlinearFactorGraph& graph() const { return graph_; }
  /// \brief Access the current estimate.
  const gtsam::Values& values() const { return values_; }

 private:
  okvis::ImuParameters imuParameters_;
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> imuParams_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;

  std::set<std::uint64_t> states_;       ///< StateIds with variables added.
  std::set<std::uint64_t> landmarks_;    ///< LandmarkIds with variables added.
  std::set<std::size_t> extrinsics_;     ///< Camera ids with extrinsics added.

  // --- delayed marginalization state ---
  /// \brief Mirror of the raw (relinearizable) factors, for re-marginalization.
  gtsam_backend::DelayedGraph delayed_;
  int delayedLag_ = 0;                          ///< Lag in keyframes; 0 disables.
  std::deque<gtsam::KeyVector> droppedBatches_; ///< Per-marginalization dropped keys.
  gtsam::KeyVector retainedDroppedKeys_;        ///< Dropped keys still retained in delayed_.
  std::vector<gtsam::NonlinearFactor::shared_ptr> activePriors_;  ///< Current marg priors in graph_.
  Eigen::Matrix<double, 6, 1> biasAtLastRemarg_ = Eigen::Matrix<double, 6, 1>::Zero();
  bool haveRemargBias_ = false;
  double lastRemargTimeSec_ = -1e18;
  int remargThrottleHits_ = 0;
  bool remargDisabled_ = false;

  /// \brief Append a raw factor to both the active and delayed graphs.
  void addRawFactor(const gtsam::NonlinearFactor::shared_ptr& factor);
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAMBACKEND_HPP_
