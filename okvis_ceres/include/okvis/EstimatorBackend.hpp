/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file EstimatorBackend.hpp
 * @brief Pulls in the concrete backend that okvis::Estimator resolves to. Include
 *        this (rather than ViSlamBackend.hpp directly) wherever the full Estimator
 *        type is needed, so the compile-time backend selection
 *        (OKVIS_USE_GTSAM_BACKEND, track-5) takes effect. Default: ViSlamBackend.
 */

#ifndef INCLUDE_OKVIS_ESTIMATORBACKEND_HPP_
#define INCLUDE_OKVIS_ESTIMATORBACKEND_HPP_

#ifdef OKVIS_USE_GTSAM_BACKEND
#include <okvis/GtsamBackend.hpp>
#else
#include <okvis/ViSlamBackend.hpp>
#endif

#endif  // INCLUDE_OKVIS_ESTIMATORBACKEND_HPP_
