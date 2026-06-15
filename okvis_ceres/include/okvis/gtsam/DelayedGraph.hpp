/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file DelayedGraph.hpp
 * @brief DM-VIO-style delayed marginalization graph (Phase A).
 *
 * Retains the *raw* (relinearizable) factors of the lag window plus a base
 * marginalization prior summarizing everything older. Two operations:
 *
 *  - advance(keysToDrop): marginalize the oldest keys out, collapsing their
 *    factors into the prior (the normal sliding-window step).
 *  - computeMarginalPrior(keysToKeep, correctedValues): re-derive a marginal
 *    prior over the active keys at a *corrected* linearization, WITHOUT mutating
 *    the retained graph. Because the raw factors are still present, this yields
 *    a prior that could not have been produced by relinearizing an already
 *    collapsed prior — this is DM-VIO's "marginalization replacement".
 *
 * The active/realtime graph marginalizes aggressively; this delayed graph lags
 * by N keyframes so it can readvance after IMU init / bias-drift corrections.
 */

#ifndef INCLUDE_OKVIS_GTSAM_DELAYEDGRAPH_HPP_
#define INCLUDE_OKVIS_GTSAM_DELAYEDGRAPH_HPP_

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

namespace okvis {
namespace gtsam_backend {

/// \brief A graph that delays marginalization and supports re-derivation.
class DelayedGraph {
 public:
  DelayedGraph() = default;

  /// \brief Append raw factors to the retained graph.
  void addFactors(const gtsam::NonlinearFactorGraph& factors);

  /// \brief Append a single raw factor.
  void addFactor(const gtsam::NonlinearFactor::shared_ptr& factor);

  /// \brief Insert values for keys not already present.
  void addValues(const gtsam::Values& values);

  /// \brief Update (overwrite) values for keys already present, insert otherwise.
  void updateValues(const gtsam::Values& values);

  const gtsam::NonlinearFactorGraph& graph() const { return graph_; }
  const gtsam::Values& values() const { return values_; }
  std::size_t size() const { return graph_.size(); }

  /// \brief Marginalize keysToDrop out, folding the connected factors into a
  ///        single relinearizable prior (mutates the retained graph/values).
  /// \return The new prior factor (or nullptr if nothing to marginalize).
  gtsam::NonlinearFactor::shared_ptr advance(const gtsam::KeyVector& keysToDrop);

  /// \brief Re-derive a marginal prior over keysToKeep at a corrected
  ///        linearization, marginalizing out every other key. Non-mutating.
  /// \param keysToKeep      The active (separator) keys to retain.
  /// \param correctedValues Linearization point for ALL keys in the graph.
  /// \return The marginal prior over keysToKeep (or nullptr).
  gtsam::NonlinearFactor::shared_ptr computeMarginalPrior(
      const gtsam::KeyVector& keysToKeep,
      const gtsam::Values& correctedValues) const;

 private:
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
};

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_DELAYEDGRAPH_HPP_
