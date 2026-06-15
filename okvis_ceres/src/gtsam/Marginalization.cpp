/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file Marginalization.cpp
 * @brief Implementation of GTSAM Schur-complement marginalization.
 */

#include <okvis/gtsam/Marginalization.hpp>

#include <boost/make_shared.hpp>

#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

namespace okvis {
namespace gtsam_backend {

gtsam::NonlinearFactor::shared_ptr marginalizeOut(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& linPoint,
    const gtsam::KeyVector& keysToDrop) {
  if (graph.empty() || keysToDrop.empty()) {
    return nullptr;
  }

  // Linearize the connected factors at the current estimate.
  const gtsam::GaussianFactorGraph::shared_ptr gfg = graph.linearize(linPoint);

  // Eliminate (marginalize) the dropped keys; the second element of the result
  // is the remaining factor graph over the separator = the Schur-complement
  // marginal.
  const gtsam::Ordering dropOrdering(keysToDrop);
  const auto result =
      gfg->eliminatePartialMultifrontal(dropOrdering, gtsam::EliminateCholesky);
  const gtsam::GaussianFactorGraph::shared_ptr marginal = result.second;
  if (!marginal || marginal->empty()) {
    return nullptr;
  }

  // Collapse the separator factors into a single dense HessianFactor and wrap it
  // as a relinearizable nonlinear prior anchored at the current linearization.
  const gtsam::HessianFactor::shared_ptr hessian =
      boost::make_shared<gtsam::HessianFactor>(*marginal);
  return boost::make_shared<gtsam::LinearContainerFactor>(hessian, linPoint);
}

}  // namespace gtsam_backend
}  // namespace okvis
