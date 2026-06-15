/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file Marginalization.hpp
 * @brief GTSAM Schur-complement marginalization producing a relinearizable prior.
 *
 * Marginalizes a set of variables out of (the factors touching them in) a
 * nonlinear graph, returning a single gtsam::LinearContainerFactor over the
 * separator. Unlike OKVIS's destructive MST pose-graph conversion, this is a
 * true marginalization prior that can be relinearized (the basis for DM-VIO's
 * delayed marginalization / marginalization replacement in later phases).
 */

#ifndef INCLUDE_OKVIS_GTSAM_MARGINALIZATION_HPP_
#define INCLUDE_OKVIS_GTSAM_MARGINALIZATION_HPP_

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

namespace okvis {
namespace gtsam_backend {

/// \brief Marginalize keysToDrop out of `graph`, linearized at `linPoint`.
/// \param graph     Factors connected to the dropped keys (the factors that will
///                  be consumed by marginalization).
/// \param linPoint  Linearization point (must contain all keys in `graph`).
/// \param keysToDrop Keys to marginalize out.
/// \return A LinearContainerFactor over the separator keys, or nullptr if the
///         marginalization is empty / fails.
gtsam::NonlinearFactor::shared_ptr marginalizeOut(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& linPoint,
    const gtsam::KeyVector& keysToDrop);

}  // namespace gtsam_backend
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GTSAM_MARGINALIZATION_HPP_
