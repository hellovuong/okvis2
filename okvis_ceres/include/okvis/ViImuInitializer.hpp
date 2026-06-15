/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file ViImuInitializer.hpp
 * @brief DM-VIO-style dynamic IMU initialization via PGBA (Phase C).
 *
 * State machine STATIC / VISUAL_ONLY / JOINT_INIT / CONVERGED that recovers
 * gravity direction (R_wg), IMU biases and per-keyframe velocities once enough
 * excitation is observed, given metric (stereo) visual poses. The joint init is
 * a small GTSAM problem combining GravityAlignmentFactor over the window with
 * the keyframe poses held fixed (Pose-Graph Bundle Adjustment for the inertial
 * unknowns). The recovered alignment/bias/velocities are returned to the caller
 * to apply atomically to the backend (rewriteAfterInit — part of the live-system
 * integration, handled separately).
 *
 * For STEREO the metric scale is observed, so scale is NOT a free variable.
 */

#ifndef INCLUDE_OKVIS_VIIMUINITIALIZER_HPP_
#define INCLUDE_OKVIS_VIIMUINITIALIZER_HPP_

#include <deque>
#include <map>

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

/// \brief Dynamic IMU initializer (DM-VIO style, stereo).
class ViImuInitializer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Initialization state.
  enum class State { Static, VisualOnly, JointInit, Converged };

  /// \brief Result of the initialization.
  struct Result {
    bool converged = false;                ///< Whether init has converged.
    bool usedStaticFallback = false;       ///< True if the static fallback fired.
    gtsam::Rot3 R_wg;                      ///< Gravity-alignment rotation (W<-G).
    gtsam::imuBias::ConstantBias bias;     ///< Recovered IMU bias.
    std::map<std::uint64_t, Eigen::Vector3d> velocities;  ///< Per-StateId v_W.
    std::map<std::uint64_t, okvis::kinematics::Transformation> poses;  ///< PGBA-refined poses (visual world).
    bool usedPgba = false;                 ///< True if true PGBA (visual prior) was used.
    double conditionNumber = 0.0;          ///< Smallest eigenvalue of the gate.
  };

  /// \brief Construct.
  explicit ViImuInitializer(const okvis::ImuParameters& imuParameters);

  /// \brief Register a keyframe with its (metric) visual pose and the IMU
  ///        measurements since the previous keyframe.
  /// \param id        State id of the keyframe.
  /// \param T_WS      Visual-only metric pose (in the arbitrary visual world).
  /// \param imuSincePrev IMU measurements spanning [tPrev, tCurr].
  /// \param tPrev     Timestamp of the previous keyframe (ignored for first).
  /// \param tCurr     Timestamp of this keyframe.
  void addKeyframe(StateId id, const okvis::kinematics::Transformation& T_WS,
                   const okvis::ImuMeasurementDeque& imuSincePrev,
                   const okvis::Time& tPrev, const okvis::Time& tCurr);

  /// \brief Provide a visual marginalization prior over the window pose keys
  ///        (keyed by okvis::gtsam_backend::poseKey(stateId)). When set, the
  ///        joint init runs TRUE PGBA: poses are free variables constrained by
  ///        this prior plus the gravity-variable IMU factors (DM-VIO). Without
  ///        it, the init falls back to inertial-only alignment (fixed poses).
  void setVisualMarginalizationPrior(
      const gtsam::NonlinearFactor::shared_ptr& prior) {
    visualPrior_ = prior;
  }

  /// \brief Advance the state machine; runs PGBA when ready.
  /// \return The current result (Result::converged set when done).
  Result step();

  /// \brief Current state.
  State state() const { return state_; }

 private:
  struct Frame {
    StateId id;
    okvis::kinematics::Transformation T_WS;
    okvis::Time timestamp;
    // Preintegration from the previous keyframe to this one (null for the first).
    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements> pim;
  };

  /// \brief Check excitation (gyro/accel variance + visual translation).
  bool excitationSufficient() const;

  /// \brief Run the PGBA inertial-only optimization over the window.
  /// \return True on success (with result populated), false if gating failed.
  bool runJointInit(Result* result) const;

  /// \brief Build the static-init result (gravity from accel, zero velocity).
  Result staticFallback() const;

  /// \brief Initial gravity-alignment guess from the mean specific force.
  gtsam::Rot3 initialGravityAlignment() const;

  okvis::ImuParameters imuParameters_;
  gtsam::NonlinearFactor::shared_ptr visualPrior_;  ///< Optional visual marg prior (PGBA).
  State state_ = State::Static;
  std::deque<Frame> window_;       ///< Recent keyframes (raw acc/gyr kept in pim).
  std::deque<Eigen::Vector3d> gyrSamples_;  ///< Recent gyro samples (excitation).
  std::deque<Eigen::Vector3d> accSamples_;  ///< Recent accel samples (excitation).
  okvis::Time startTime_;          ///< Time of the first keyframe.
  bool started_ = false;
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_VIIMUINITIALIZER_HPP_
