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

#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/MultiFrame.hpp>
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

  /// \brief Default constructor (IMU params set later via addImu).
  GtsamBackend();
  /// \brief Construct with IMU parameters (for preintegration noise / gravity).
  explicit GtsamBackend(const okvis::ImuParameters& imuParameters);

  // --- sensors (Estimator API, track-5 S1) ----------------------------------
  /// \brief Register IMU parameters (rebuilds the preintegration params). Index 0.
  int addImu(const okvis::ImuParameters& imuParameters);
  /// \brief Register a camera's online-calibration parameters. Returns its index.
  int addCamera(const okvis::CameraParameters& cameraParameters);

  // --- per-frame state creation (Estimator API, track-5 S1) ------------------
  /// \brief Add a new frame's states: initialise (first frame, gravity-aligned)
  ///        or IMU-propagate from the previous state and link with an IMU factor.
  bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque& imuMeasurements,
                 bool asKeyframe);
  /// \brief Core of addStates, decoupled from MultiFrame for testing.
  bool addPropagatedState(StateId id, const okvis::Time& timestamp,
                          const okvis::ImuMeasurementDeque& imuMeasurements,
                          bool asKeyframe);

  // --- extrinsics -----------------------------------------------------------
  /// \brief Add/define camera extrinsics T_SC. If fixed, anchors with a tight
  ///        prior (online calibration releases this later).
  void setExtrinsics(std::size_t cameraId,
                     const okvis::kinematics::Transformation& T_SC,
                     bool fixed = true);

  // --- states ---------------------------------------------------------------
  /// \brief Insert a state (pose + velocity + bias) as initial values.
  /// \param timestamp  State timestamp (for the Estimator API; default 0).
  /// \param isKeyframe Whether this state is a keyframe.
  void addState(StateId id, const okvis::kinematics::Transformation& T_WS,
                const okvis::SpeedAndBias& speedAndBias,
                const okvis::Time& timestamp = okvis::Time(0),
                bool isKeyframe = false);

  // --- Estimator-API state/window queries (track-5 S1) ----------------------
  /// \brief Timestamp of a state (or invalid Time if unknown).
  okvis::Time timestamp(StateId id) const;
  /// \brief Whether a state is a keyframe.
  bool isKeyframe(StateId id) const;
  /// \brief Flag/unflag a state as keyframe.
  void setKeyframe(StateId id, bool isKeyframe);
  /// \brief The most-recent (highest-id) state, or invalid if none.
  StateId currentStateId() const;
  /// \brief State by age: age 0 = newest, 1 = next newest, ... (invalid if OOB).
  StateId stateIdByAge(std::size_t age) const;
  /// \brief Number of states currently in the window.
  std::size_t numFrames() const { return states_.size(); }
  /// \brief Whether a state is in the (recent) IMU window.
  bool isInImuWindow(StateId id) const { return imuFrames_.count(id) > 0; }
  /// \brief Current keyframes.
  const std::set<StateId>& keyFrames() const { return keyFrames_; }
  /// \brief Current IMU-window frames.
  const std::set<StateId>& imuFrames() const { return imuFrames_; }

  // --- window strategy + optimisation entry (Estimator API, track-5 S1) ------
  /// \brief Sliding-window policy: keep the newest `numImuFrames` states plus the
  ///        newest `numKeyframes` keyframes; Schur-marginalize the rest (oldest
  ///        first). `numLoopClosureFrames` is ignored until S3 (loop closure).
  bool applyStrategy(std::size_t numKeyframes, std::size_t numLoopClosureFrames,
                     std::size_t numImuFrames, std::set<StateId>& affectedFrames,
                     bool expand = true);
  /// \brief Optimise the (single) realtime graph; fills updatedStates with all
  ///        states. (numThreads/onlyNewestState/isInitialised reserved for parity.)
  void optimiseRealtimeGraph(int numIter, std::vector<StateId>& updatedStates,
                             int numThreads = 1, bool verbose = false,
                             bool onlyNewestState = false, bool isInitialised = true);
  /// \brief Store an optimisation time budget (batch LM honours minIterations).
  bool setOptimisationTimeLimit(double timeLimit, int minIterations);

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
  bool addLandmark(LandmarkId id, const Eigen::Vector4d& hp_W,
                   bool initialised = false);
  /// \brief Insert a landmark with an auto-assigned id; returns the new id.
  LandmarkId addLandmark(const Eigen::Vector4d& hp_W, bool initialised) {
    const LandmarkId id(nextLandmarkId_++);
    addLandmark(id, hp_W, initialised);
    return id;
  }

  // --- landmark metadata (Estimator API, track-5 S2) ------------------------
  /// \brief Set a landmark's world position and initialisation flag.
  bool setLandmark(LandmarkId id, const Eigen::Vector4d& hp_W, bool isInitialised);
  /// \brief Set a landmark's initialisation flag.
  bool setLandmarkInitialized(LandmarkId id, bool initialised);
  /// \brief Set a landmark's classification id.
  bool setLandmarkClassification(LandmarkId id, int classification);
  /// \brief Whether the landmark exists in the graph.
  bool isLandmarkAdded(LandmarkId id) const { return landmarks_.count(id.value()) > 0; }
  /// \brief Whether the landmark is flagged initialised.
  bool isLandmarkInitialised(LandmarkId id) const;
  /// \brief Fill a MapPoint2 with the landmark estimate + metadata.
  bool getLandmark(LandmarkId id, okvis::MapPoint2& mapPoint) const;

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

  // --- data association (Estimator API, track-5 S2) -------------------------
  /// \brief Add a tracked observation (keyed by KeypointIdentifier) with full
  ///        bookkeeping so it can later be queried/removed/cleaned. The factor is
  ///        tracked by index in the active graph for O(1) removal.
  template <class GEOMETRY_TYPE>
  bool addObservation(LandmarkId landmarkId, KeypointIdentifier kid,
                      const Eigen::Vector2d& measurement,
                      const Eigen::Matrix2d& information,
                      std::shared_ptr<const GEOMETRY_TYPE> cameraGeometry,
                      bool useCauchy = true) {
    if (landmarks_.count(landmarkId.value()) == 0) return false;
    if (observations_.count(kid)) return false;
    typedef okvis::gtsam_backend::GtsamReprojectionFactor<GEOMETRY_TYPE> Factor;
    const double cauchy = useCauchy ? cauchyParam_ : 0.0;
    const gtsam::NonlinearFactor::shared_ptr factor = boost::make_shared<Factor>(
        Factor::makeNoiseModel(information, cauchy),
        okvis::gtsam_backend::poseKey(kid.frameId),
        okvis::gtsam_backend::landmarkKey(landmarkId.value()),
        okvis::gtsam_backend::extrinsicsKey(kid.cameraIndex), measurement,
        std::move(cameraGeometry));
    graph_.push_back(factor);
    observations_[kid] = ObsRecord{landmarkId, factor};
    landmarkObs_[landmarkId.value()].insert(kid);
    return true;
  }

  /// \brief Add an observation, extracting measurement + info (64/size^2) and the
  ///        camera geometry from the MultiFrame (the live frontend signature).
  template <class GEOMETRY_TYPE>
  bool addObservation(const okvis::MultiFrame& multiFrame, LandmarkId landmarkId,
                      KeypointIdentifier kid, bool useCauchy = true) {
    Eigen::Vector2d measurement;
    multiFrame.getKeypoint(kid.cameraIndex, kid.keypointIndex, measurement);
    double size = 1.0;
    multiFrame.getKeypointSize(kid.cameraIndex, kid.keypointIndex, size);
    const Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * (64.0 / (size * size));
    return addObservation<GEOMETRY_TYPE>(
        landmarkId, kid, measurement, info,
        multiFrame.template geometryAs<GEOMETRY_TYPE>(kid.cameraIndex), useCauchy);
  }

  /// \brief Add an observation by (landmark, state, camera, keypoint) — the live
  ///        frontend signature; looks up the stored multiframe and delegates.
  template <class GEOMETRY_TYPE>
  bool addObservation(LandmarkId landmarkId, StateId stateId, std::size_t camIdx,
                      std::size_t keypointIdx, bool useCauchy = true) {
    const auto it = multiFrames_.find(stateId);
    if (it == multiFrames_.end() || !it->second) return false;
    return addObservation<GEOMETRY_TYPE>(
        *it->second, landmarkId,
        KeypointIdentifier(stateId.value(), camIdx, keypointIdx), useCauchy);
  }

  /// \brief Whether a keypoint observation is present.
  bool isObserved(KeypointIdentifier kid) const { return observations_.count(kid) > 0; }
  /// \brief Remove a tracked observation (drops its factor from the active graph).
  bool removeObservation(KeypointIdentifier kid);
  /// \brief Remove an observation by (state, camera, keypoint).
  bool removeObservation(StateId stateId, std::size_t camIdx, std::size_t keypointIdx) {
    return removeObservation(KeypointIdentifier(stateId.value(), camIdx, keypointIdx));
  }
  /// \brief Remove landmarks that have no remaining observations. Returns the count.
  int cleanUnobservedLandmarks();

  /// \brief States co-observing a landmark with `id` (covisibility), excluding id.
  bool getObservedIds(StateId id, std::set<StateId>& observedIds) const;
  /// \brief Fill all current landmarks as MapPoint2 (id -> point/init/quality/class).
  std::size_t getLandmarks(okvis::MapPoints& landmarks) const;

  // --- loop-closure / full-graph: no-op stubs (S3 will implement) -----------
  // Frontend/ThreadedSlam query these; with loop closure disabled they are inert.
  bool isLoopClosing() const { return false; }
  bool isLoopClosureAvailable() const { return false; }
  bool isLoopClosureFrame(StateId) const { return false; }
  bool isRecentLoopClosureFrame(StateId) const { return false; }
  bool isPlaceRecognitionFrame(StateId) const { return false; }
  bool isPoseGraphFrame(StateId) const { return false; }
  bool closedLoop(StateId) const { return false; }
  bool needsFullGraphOptimisation() const { return false; }
  const std::set<StateId>& loopClosureFrames() const { return loopClosureFrames_; }

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

  // --- Estimator-API getters/setters + stubs (track-5 integration) ----------
  /// \brief Pose getter (const ref; backed by a lazily-updated cache).
  const okvis::kinematics::TransformationCacheless& pose(StateId id) const;
  /// \brief Speed/bias getter (const ref; cached).
  const okvis::SpeedAndBias& speedAndBias(StateId id) const;
  /// \brief Extrinsics getter (const ref; cached, per camera).
  const okvis::kinematics::TransformationCacheless& extrinsics(StateId id,
                                                               unsigned char camIdx) const;
  /// \brief Multiframe accessor (nullptr if not stored).
  okvis::MultiFramePtr multiFrame(StateId stateId) const;

  bool setPose(StateId id, const okvis::kinematics::TransformationCacheless& pose);
  bool setSpeedAndBias(StateId id, const okvis::SpeedAndBias& speedAndBias);
  bool setObservationInformation(StateId stateId, std::size_t camIdx,
                                 std::size_t keypointIdx,
                                 const Eigen::Matrix2d& information);
  void setDetectorUniformityRadius(double /*uniformityRadius*/) {}
  double trackingQuality(StateId /*id*/) const { return 1.0; }

  /// \brief Apply dynamic IMU init (rotate world by R_gw=q_gw, set bias). Wraps
  ///        the ViGraphEstimator-style rewrite onto this backend.
  bool applyInitialisation(const Eigen::Quaterniond& q_gw,
                           const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a);

  bool mergeLandmark(const LandmarkId& fromId, const LandmarkId& intoId);
  int mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds);
  bool areLandmarksInFrontOfCameras() const { return true; }

  // Loop-closure / full-graph / map: stubs (S3/S4 will implement).
  StateId mostOverlappedStateId(StateId /*frame*/, bool = true) const { return StateId(); }
  double overlapFraction(const okvis::MultiFramePtr, const okvis::MultiFramePtr) const {
    return 0.0;
  }
  bool attemptLoopClosure(StateId, StateId, const okvis::kinematics::Transformation&,
                          const Eigen::Matrix<double, 6, 6>&,
                          bool& skipFullGraphOptimisation, double) {
    skipFullGraphOptimisation = true;
    return false;
  }
  void addLoopClosureFrame(StateId, std::set<LandmarkId>&, bool) {}
  void optimiseFullGraph(int, ::ceres::Solver::Summary&, int = 1, bool = false) {}
  void doFinalBa(int numIter, ::ceres::Solver::Summary& summary,
                 std::set<StateId>& updatedStatesBa, double = 0.0, double = 0.0,
                 int = 1, bool = false);
  bool synchroniseRealtimeAndFullGraph(std::vector<StateId>&) { return false; }
  bool saveMap(std::string /*path*/) { return false; }
  bool writeFinalCsvTrajectory(const std::string& csvFileName, bool rpg = false) const;
  void drawOverheadImage(cv::Mat& /*image*/, int = 0) const {}
  void clear();

  /// \brief Per-state anchor transforms (loop-closure/multi-session). Empty stub.
  okvis::AlignedMap<StateId, okvis::AlignedMap<std::uint64_t,
                    okvis::kinematics::Transformation>> T_AiS_;

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

  /// \brief Per-state metadata for the Estimator API.
  struct StateMeta {
    okvis::Time timestamp;
    bool isKeyframe = false;
  };
  std::map<std::uint64_t, StateMeta> stateMeta_;  ///< Keyed by StateId value.
  std::vector<okvis::CameraParameters,
              Eigen::aligned_allocator<okvis::CameraParameters>> cameraParams_;  ///< Registered cameras.
  std::set<StateId> keyFrames_;   ///< Current keyframes.
  std::set<StateId> imuFrames_;   ///< Current IMU-window frames.
  std::set<StateId> loopClosureFrames_;  ///< Loop-closure frames (always empty until S3).

  /// \brief Per-landmark metadata for the Estimator API.
  struct LandmarkMeta {
    bool initialised = false;
    int classification = -1;
    double quality = 0.0;
  };
  std::map<std::uint64_t, LandmarkMeta> landmarkMeta_;  ///< Keyed by LandmarkId value.

  /// \brief Tracked observation record (for query/removal). Stores the factor
  ///        pointer (stable across graph rebuilds, unlike an index).
  struct ObsRecord {
    LandmarkId landmarkId;
    gtsam::NonlinearFactor::shared_ptr factor;
  };
  std::map<KeypointIdentifier, ObsRecord> observations_;          ///< By keypoint id.
  std::map<std::uint64_t, std::set<KeypointIdentifier>> landmarkObs_;  ///< Per-landmark obs.
  double cauchyParam_ = 1.0;  ///< Reprojection Cauchy robustifier scale.
  std::uint64_t nextLandmarkId_ = 1;  ///< Counter for auto-assigned landmark ids.
  okvis::AlignedMap<StateId, okvis::MultiFramePtr> multiFrames_;  ///< Stored multiframes.

  // Caches backing the const-reference Estimator getters (computed from Values).
  mutable std::map<std::uint64_t, okvis::kinematics::TransformationCacheless> poseCache_;
  mutable std::map<std::uint64_t, okvis::SpeedAndBias> sbCache_;
  mutable std::map<std::size_t, okvis::kinematics::TransformationCacheless> extrinsicsCache_;
  double optTimeLimit_ = -1.0;    ///< Optimisation time budget [s] (<0: none).
  int optMinIterations_ = 3;      ///< Minimum LM iterations regardless of budget.

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
