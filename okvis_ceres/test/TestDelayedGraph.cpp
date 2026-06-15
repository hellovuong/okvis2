/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestDelayedGraph.cpp
 * @brief Validates DelayedGraph: advance() yields the correct marginal, and
 *        computeMarginalPrior() re-derives the marginal at a corrected
 *        linearization (DM-VIO marginalization replacement).
 */

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <okvis/gtsam/DelayedGraph.hpp>

using gtsam::symbol_shorthand::X;

namespace {

// A 3-pose chain with a tight prior on X(1) and rotating between-factors (so the
// factors are genuinely nonlinear in the poses).
gtsam::NonlinearFactorGraph makeChain(gtsam::Values* trueValues) {
  const gtsam::Pose3 p1(gtsam::Rot3(), gtsam::Point3(0, 0, 0));
  const gtsam::Pose3 p2(gtsam::Rot3::Rz(0.3), gtsam::Point3(1, 0.2, 0));
  const gtsam::Pose3 p3(gtsam::Rot3::Rz(0.6), gtsam::Point3(2, 0.5, 0.1));
  trueValues->insert(X(1), p1);
  trueValues->insert(X(2), p2);
  trueValues->insert(X(3), p3);

  const auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(6, 0.01);
  const auto betweenNoise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  gtsam::NonlinearFactorGraph g;
  g.addPrior(X(1), p1, priorNoise);
  g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(1), X(2), p1.between(p2),
                                                       betweenNoise);
  g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(2), X(3), p2.between(p3),
                                                       betweenNoise);
  return g;
}

}  // namespace

TEST(DelayedGraph, AdvancePreservesMarginal) {
  gtsam::Values trueValues;
  const gtsam::NonlinearFactorGraph full = makeChain(&trueValues);

  const gtsam::Marginals margFull(full, trueValues);
  const Eigen::MatrixXd cov3Full = margFull.marginalCovariance(X(3));

  okvis::gtsam_backend::DelayedGraph dg;
  dg.addFactors(full);
  dg.addValues(trueValues);

  // Marginalize out the oldest pose.
  const auto prior = dg.advance({X(1)});
  ASSERT_TRUE(prior != nullptr);
  EXPECT_FALSE(dg.values().exists(X(1)));

  const gtsam::Marginals margDg(dg.graph(), dg.values());
  const Eigen::MatrixXd cov3Dg = margDg.marginalCovariance(X(3));
  EXPECT_LT((cov3Full - cov3Dg).norm(), 1e-6 * cov3Full.norm());
}

TEST(DelayedGraph, ComputeMarginalPriorMatchesFull) {
  gtsam::Values trueValues;
  const gtsam::NonlinearFactorGraph full = makeChain(&trueValues);

  const gtsam::Marginals margFull(full, trueValues);
  const Eigen::MatrixXd cov3Full = margFull.marginalCovariance(X(3));

  okvis::gtsam_backend::DelayedGraph dg;
  dg.addFactors(full);
  dg.addValues(trueValues);

  // Non-mutating marginal over X(3) at the true linearization.
  const auto prior = dg.computeMarginalPrior({X(3)}, trueValues);
  ASSERT_TRUE(prior != nullptr);

  gtsam::NonlinearFactorGraph priorGraph;
  priorGraph.push_back(prior);
  gtsam::Values v3;
  v3.insert(X(3), trueValues.at<gtsam::Pose3>(X(3)));
  const gtsam::Marginals margPrior(priorGraph, v3);
  const Eigen::MatrixXd cov3Prior = margPrior.marginalCovariance(X(3));

  EXPECT_LT((cov3Full - cov3Prior).norm(), 1e-6 * cov3Full.norm());

  // The retained graph is untouched by computeMarginalPrior.
  EXPECT_TRUE(dg.values().exists(X(1)));
  EXPECT_TRUE(dg.values().exists(X(2)));
}

TEST(DelayedGraph, ReadvanceRelinearizes) {
  gtsam::Values trueValues;
  const gtsam::NonlinearFactorGraph full = makeChain(&trueValues);

  okvis::gtsam_backend::DelayedGraph dg;
  dg.addFactors(full);
  dg.addValues(trueValues);

  // Marginal over X(3) at the true linearization.
  const auto priorTrue = dg.computeMarginalPrior({X(3)}, trueValues);
  ASSERT_TRUE(priorTrue != nullptr);

  // Re-derive at a corrected/perturbed linearization (e.g. after a bias update
  // shifts the intermediate poses). Because the raw factors are retained, this
  // produces a genuinely different marginal.
  gtsam::Values corrected = trueValues;
  corrected.update(X(1), gtsam::Pose3(gtsam::Rot3::Rz(0.25),
                                      gtsam::Point3(0.1, -0.1, 0.05)));
  corrected.update(X(2), gtsam::Pose3(gtsam::Rot3::Rz(0.9),
                                      gtsam::Point3(1.2, 0.4, -0.1)));
  const auto priorCorrected = dg.computeMarginalPrior({X(3)}, corrected);
  ASSERT_TRUE(priorCorrected != nullptr);

  // Compare the two marginals on X(3): relinearization must change the prior.
  gtsam::Values v3;
  v3.insert(X(3), trueValues.at<gtsam::Pose3>(X(3)));

  gtsam::NonlinearFactorGraph gTrue;
  gTrue.push_back(priorTrue);
  gtsam::NonlinearFactorGraph gCorr;
  gCorr.push_back(priorCorrected);

  const Eigen::MatrixXd covTrue = gtsam::Marginals(gTrue, v3).marginalCovariance(X(3));
  const Eigen::MatrixXd covCorr = gtsam::Marginals(gCorr, v3).marginalCovariance(X(3));

  EXPECT_GT((covTrue - covCorr).norm(), 1e-4)
      << "relinearization should change the marginal";
}
