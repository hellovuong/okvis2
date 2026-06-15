/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use permitted under the OKVIS BSD-3-Clause license.
 *********************************************************************************/

/**
 * @file TestCeresResidualSwap.cpp
 * @brief Track-4 milestone 1 (gating prototype): confirm that the marginalization
 *        prior factor can be SWAPPED mid-run — remove a relative-pose factor and
 *        add a re-derived one between Solve() calls — and the problem re-converges
 *        correctly with OKVIS's parameter blocks / manifold / solver options.
 *
 * This is the exact mechanic delayed-marginalization replacement needs (rebuild a
 * TwoPoseGraphError from preserved observations at corrected linearization and
 * swap it in). OKVIS already does Remove+Add of residual blocks between solves as
 * its normal marginalization mechanism (enable_fast_removal=true); this pins it
 * down as a standalone, reproducible check.
 */

#include <gtest/gtest.h>

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>

TEST(CeresResidualSwap, RemarginalizeSwapReconverges) {
  // Problem options matching ViGraph (enable_fast_removal; external ownership).
  ::ceres::Problem::Options opts;
  opts.manifold_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
  opts.cost_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
  opts.loss_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
  opts.enable_fast_removal = true;
  ::ceres::Problem problem(opts);

  okvis::ceres::PoseManifold poseManifold;

  const okvis::kinematics::Transformation T_WA;  // identity (anchor)
  okvis::kinematics::Transformation T_WB_init(Eigen::Vector3d(0.3, 0.0, 0.0),
                                              Eigen::Quaterniond::Identity());
  okvis::ceres::PoseParameterBlock poseA(T_WA, 1u, okvis::Time(0));
  okvis::ceres::PoseParameterBlock poseB(T_WB_init, 2u, okvis::Time(0));
  problem.AddParameterBlock(poseA.parameters(), 7, &poseManifold);
  problem.AddParameterBlock(poseB.parameters(), 7, &poseManifold);

  // Strong prior pinning A at identity.
  Eigen::Matrix<double, 6, 1> infoDiag;
  infoDiag.setConstant(1.0e8);
  okvis::ceres::PoseError priorA(T_WA, infoDiag);
  problem.AddResidualBlock(&priorA, nullptr, poseA.parameters());

  const Eigen::Matrix<double, 6, 6> info =
      Eigen::Matrix<double, 6, 6>::Identity() * 1.0e4;

  // Initial marginalization-like relative factor: B = A * T_AB1.
  const okvis::kinematics::Transformation T_AB1(Eigen::Vector3d(1.0, 0.0, 0.0),
                                                Eigen::Quaterniond::Identity());
  okvis::ceres::RelativePoseError rel1(info, T_AB1);
  const ::ceres::ResidualBlockId rid1 = problem.AddResidualBlock(
      &rel1, nullptr, poseA.parameters(), poseB.parameters());

  ::ceres::Solver::Options sopts;
  sopts.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
  sopts.max_num_iterations = 30;
  ::ceres::Solver::Summary summary;
  ::ceres::Solve(sopts, &problem, &summary);
  ASSERT_TRUE(summary.IsSolutionUsable());
  EXPECT_LT((poseB.estimate().r() - Eigen::Vector3d(1.0, 0.0, 0.0)).norm(), 1e-3);

  // --- marginalization replacement: swap the relative factor for a re-derived
  //     one (different relative measurement, as after a bias correction) ---
  problem.RemoveResidualBlock(rid1);
  const okvis::kinematics::Transformation T_AB2(
      Eigen::Vector3d(2.0, 0.5, 0.0),
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())));
  okvis::ceres::RelativePoseError rel2(info, T_AB2);
  problem.AddResidualBlock(&rel2, nullptr, poseA.parameters(), poseB.parameters());

  ::ceres::Solve(sopts, &problem, &summary);
  ASSERT_TRUE(summary.IsSolutionUsable());

  // B must re-converge to the NEW relative constraint (A is at identity -> B≈T_AB2).
  EXPECT_LT((poseB.estimate().r() - T_AB2.r()).norm(), 1e-3)
      << "B did not follow the swapped factor; got " << poseB.estimate().r().transpose();
  EXPECT_NEAR(
      std::abs(Eigen::Quaterniond(poseB.estimate().q()).dot(Eigen::Quaterniond(T_AB2.q()))),
      1.0, 1e-3);
  // A stayed anchored at identity.
  EXPECT_LT(poseA.estimate().r().norm(), 1e-4);
}
