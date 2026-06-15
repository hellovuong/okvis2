/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file DelayedGraph.cpp
 * @brief Implementation of the DM-VIO delayed marginalization graph.
 */

#include <okvis/gtsam/DelayedGraph.hpp>

#include <set>

#include <okvis/gtsam/Marginalization.hpp>

namespace okvis {
namespace gtsam_backend {

void DelayedGraph::addFactors(const gtsam::NonlinearFactorGraph& factors) {
  for (const auto& f : factors) {
    if (f) graph_.push_back(f);
  }
}

void DelayedGraph::addFactor(const gtsam::NonlinearFactor::shared_ptr& factor) {
  if (factor) graph_.push_back(factor);
}

void DelayedGraph::addValues(const gtsam::Values& values) {
  for (const auto key : values.keys()) {
    if (!values_.exists(key)) {
      values_.insert(key, values.at(key));
    }
  }
}

void DelayedGraph::updateValues(const gtsam::Values& values) {
  for (const auto key : values.keys()) {
    if (values_.exists(key)) {
      values_.update(key, values.at(key));
    } else {
      values_.insert(key, values.at(key));
    }
  }
}

gtsam::NonlinearFactor::shared_ptr DelayedGraph::advance(
    const gtsam::KeyVector& keysToDrop) {
  if (keysToDrop.empty()) return nullptr;
  const std::set<gtsam::Key> dropSet(keysToDrop.begin(), keysToDrop.end());

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
    (touches ? touching : keep).push_back(factor);
  }

  const gtsam::NonlinearFactor::shared_ptr prior =
      marginalizeOut(touching, values_, keysToDrop);

  graph_ = keep;
  if (prior) graph_.push_back(prior);
  for (const gtsam::Key k : keysToDrop) {
    if (values_.exists(k)) values_.erase(k);
  }
  return prior;
}

gtsam::NonlinearFactor::shared_ptr DelayedGraph::computeMarginalPrior(
    const gtsam::KeyVector& keysToKeep,
    const gtsam::Values& correctedValues) const {
  const std::set<gtsam::Key> keepSet(keysToKeep.begin(), keysToKeep.end());
  gtsam::KeyVector keysToDrop;
  for (const auto key : values_.keys()) {
    if (!keepSet.count(key)) keysToDrop.push_back(key);
  }
  if (keysToDrop.empty()) return nullptr;
  return marginalizeOut(graph_, correctedValues, keysToDrop);
}

gtsam::NonlinearFactor::shared_ptr DelayedGraph::recomputeBoundaryPrior(
    const gtsam::KeyVector& keysToDrop,
    const gtsam::Values& correctedValues) const {
  if (keysToDrop.empty()) return nullptr;
  const std::set<gtsam::Key> dropSet(keysToDrop.begin(), keysToDrop.end());

  // Only the factors that touch a dropped key contribute to the boundary prior;
  // factors over purely-live keys are left for the active graph (no double count).
  gtsam::NonlinearFactorGraph touching;
  for (const auto& factor : graph_) {
    if (!factor) continue;
    for (const gtsam::Key k : factor->keys()) {
      if (dropSet.count(k)) {
        touching.push_back(factor);
        break;
      }
    }
  }
  return marginalizeOut(touching, correctedValues, keysToDrop);
}

}  // namespace gtsam_backend
}  // namespace okvis
