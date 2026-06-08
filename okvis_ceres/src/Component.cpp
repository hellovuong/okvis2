/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab, ETH Zurich, Smart Robotics Lab,
 *     Imperial College London, Technical University of Munich, nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

/**
 * @file Component.cpp
 * @brief Source file for the Component class.
 * @author Stefan Leutenegger
 *
 * The on-disk map format is a single SQLite3 database holding the full,
 * re-optimisable visual-inertial graph (poses, speed/biases, landmarks,
 * observations, BRISK2 descriptors and the raw IMU measurements per edge).
 */


#include <cstring>
#include <iostream>

#include <sqlite3.h>

#include "okvis/cameras/EquidistantDistortion.hpp"
#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/RadialTangentialDistortion.hpp"
#include "okvis/cameras/RadialTangentialDistortion8.hpp"

#include <okvis/Component.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {

/// \brief Format version written to / expected from the meta table.
static const int kMapFormatVersion = 1;

/// \brief Run a simple SQL statement, throwing on error.
void execSql(sqlite3 *db, const char *sql) {
  char *errMsg = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
    std::string msg = errMsg ? errMsg : "unknown error";
    sqlite3_free(errMsg);
    throw Component::Exception(std::string("SQLite error: ") + msg + " (" + sql + ")");
  }
}

/// \brief RAII wrapper for a prepared statement.
class Stmt {
 public:
  Stmt(sqlite3 *db, const char *sql) : db_(db) {
    if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
      throw Component::Exception(std::string("SQLite prepare failed: ")
                                 + sqlite3_errmsg(db) + " (" + sql + ")");
    }
  }
  ~Stmt() { sqlite3_finalize(stmt_); }
  sqlite3_stmt *get() { return stmt_; }
  /// \brief Execute a step-to-completion statement (INSERT/UPDATE).
  void stepDone() {
    if (sqlite3_step(stmt_) != SQLITE_DONE) {
      throw Component::Exception(std::string("SQLite step failed: ") + sqlite3_errmsg(db_));
    }
    sqlite3_reset(stmt_);
  }
 private:
  sqlite3 *db_;
  sqlite3_stmt *stmt_ = nullptr;
};

}  // namespace

Component::Component(const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     ViGraphEstimator &fullGraph,
                     std::map<StateId, MultiFramePtr> multiFrames)
    : imuParameters_(imuParameters)
    , nCameraSystem_(nCameraSystem)
    , fullGraph_(&fullGraph)
    , multiFrames_(multiFrames)
{}

Component::Component(const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem)
    : imuParameters_(imuParameters)
    , nCameraSystem_(nCameraSystem)
{}

bool Component::load(const std::string &path)
{
  OKVIS_ASSERT_TRUE(Exception, !fullGraph_, "Graph not empty.");
  fullGraphOwn_.reset(new ViGraphEstimator());
  fullGraph_ = fullGraphOwn_.get();
  return loadInto(path, *fullGraph_, multiFrames_);
}

bool Component::loadInto(const std::string &path, ViGraphEstimator &graph,
                         std::map<StateId, MultiFramePtr> &frames, bool loadImuEdges)
{
  ViGraphEstimator *const fullGraph_ = &graph;          // local alias for the body below
  std::map<StateId, MultiFramePtr> &multiFrames_ = frames; // local alias for the body below

  // find distortion type
  cameras::NCameraSystem::DistortionType distType = nCameraSystem_.distortionType(0);
  for (size_t i = 1; i < nCameraSystem_.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distType == nCameraSystem_.distortionType(i),
                      "mixed frame types are not supported yet")
  }

  // open the database read-only
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    LOG(ERROR) << "Could not open " << path << " for reading: " << sqlite3_errmsg(db);
    sqlite3_close(db);
    return false;
  }

  try {
    // -- validate meta / descriptor type --
    {
      Stmt q(db, "SELECT value FROM meta WHERE key='descriptor_type'");
      if (sqlite3_step(q.get()) == SQLITE_ROW) {
        std::string descType =
          reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 0));
        OKVIS_ASSERT_TRUE(Exception, descType == "BRISK2",
                          "descriptor " << descType << " not supported, only BRISK 2")
      }
    }

    // -- 1. states: pose + speed/bias --
    {
      Stmt q(db, "SELECT state_id, ts_ns, tx,ty,tz, qx,qy,qz,qw, "
                 "vx,vy,vz, bgx,bgy,bgz, bax,bay,baz FROM states");
      while (sqlite3_step(q.get()) == SQLITE_ROW) {
        sqlite3_stmt *s = q.get();
        uint64_t id64 = uint64_t(sqlite3_column_int64(s, 0));
        StateId id(id64);
        okvis::Time timestamp;
        timestamp.fromNSec(uint64_t(sqlite3_column_int64(s, 1)));

        kinematics::Transformation T_WS(
          Eigen::Vector3d(sqlite3_column_double(s, 2), sqlite3_column_double(s, 3),
                          sqlite3_column_double(s, 4)),
          Eigen::Quaterniond(sqlite3_column_double(s, 8), sqlite3_column_double(s, 5),
                             sqlite3_column_double(s, 6), sqlite3_column_double(s, 7)));

        fullGraph_->states_[id] = ViGraph::State();
        ViGraph::State &state = fullGraph_->states_.at(id);
        state.timestamp = timestamp;
        state.pose.reset(new ceres::PoseParameterBlock(T_WS, id64, timestamp));
        fullGraph_->problem_->AddParameterBlock(state.pose->parameters(), 7,
                                                &fullGraph_->poseManifold_);
        state.pose->setLocalParameterizationPtr(&fullGraph_->poseManifold_);

        // speed and bias: [v(0-2), gyrbias(3-5), accbias(6-8)]
        SpeedAndBias sb;
        sb << sqlite3_column_double(s, 9), sqlite3_column_double(s, 10),
          sqlite3_column_double(s, 11), sqlite3_column_double(s, 12),
          sqlite3_column_double(s, 13), sqlite3_column_double(s, 14),
          sqlite3_column_double(s, 15), sqlite3_column_double(s, 16),
          sqlite3_column_double(s, 17);
        state.speedAndBias.reset(
          new ceres::SpeedAndBiasParameterBlock(sb, id64, timestamp));
        fullGraph_->problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);
      }
    }

    // -- 2. extrinsics / frames: create multiframes and extrinsics blocks --
    // keep track of added extrinsics (shared per extr_id, per camera)
    std::vector<std::map<uint64_t, std::shared_ptr<ceres::PoseParameterBlock>>>
      extrinsicsParameterBlocks(nCameraSystem_.numCameras());
    {
      Stmt q(db, "SELECT state_id, cam_idx, extr_id, tx,ty,tz, qx,qy,qz,qw, ts_ns "
                 "FROM extrinsics ORDER BY state_id, cam_idx");
      while (sqlite3_step(q.get()) == SQLITE_ROW) {
        sqlite3_stmt *s = q.get();
        uint64_t stateId64 = uint64_t(sqlite3_column_int64(s, 0));
        uint64_t cameraIdx = uint64_t(sqlite3_column_int64(s, 1));
        uint64_t extrId64 = uint64_t(sqlite3_column_int64(s, 2));
        kinematics::Transformation T_SC(
          Eigen::Vector3d(sqlite3_column_double(s, 3), sqlite3_column_double(s, 4),
                          sqlite3_column_double(s, 5)),
          Eigen::Quaterniond(sqlite3_column_double(s, 9), sqlite3_column_double(s, 6),
                             sqlite3_column_double(s, 7), sqlite3_column_double(s, 8)));
        Time timestamp;
        timestamp.fromNSec(uint64_t(sqlite3_column_int64(s, 10)));

        // create frame if needed
        bool firstFrame = false;
        MultiFramePtr multiFrame;
        auto iter = multiFrames_.find(StateId(stateId64));
        if (iter != multiFrames_.end()) {
          multiFrame = iter->second;
        } else {
          multiFrame.reset(new MultiFrame(nCameraSystem_, timestamp, stateId64));
          multiFrames_[StateId(stateId64)] = multiFrame;
          firstFrame = true;
        }

        ViGraph::State &state = fullGraph_->states_.at(StateId(stateId64));
        // Resize per (this) graph's state, not per multiframe: when loading the
        // same map into a second graph the multiframe is reused (firstFrame ==
        // false) but this graph's state still needs its extrinsics vector sized.
        if (state.extrinsics.size() != multiFrame->numFrames()) {
          state.extrinsics.resize(multiFrame->numFrames());
        }
        (void)firstFrame;
        if (extrinsicsParameterBlocks.at(cameraIdx).count(extrId64)) {
          state.extrinsics.at(cameraIdx) =
            extrinsicsParameterBlocks.at(cameraIdx).at(extrId64);
        } else {
          state.extrinsics.at(cameraIdx).reset(
            new ceres::PoseParameterBlock(T_SC, extrId64, timestamp));
          extrinsicsParameterBlocks.at(cameraIdx)[extrId64] = state.extrinsics.at(cameraIdx);
          fullGraph_->problem_->AddParameterBlock(
            state.extrinsics.at(cameraIdx)->parameters(), 7, &fullGraph_->poseManifold_);
        }
      }
    }

    // -- 3. keypoints + descriptors per (state, camera) --
    {
      Stmt q(db, "SELECT state_id, cam_idx, kp_idx, x, y, size, descriptor "
                 "FROM keypoints ORDER BY state_id, cam_idx, kp_idx");
      uint64_t curState = 0, curCam = 0;
      bool have = false;
      std::vector<cv::KeyPoint> keypoints;
      std::vector<cv::Mat> descriptors;
      auto flush = [&]() {
        if (!have) return;
        MultiFramePtr multiFrame = multiFrames_.at(StateId(curState));
        multiFrame->resetKeypoints(curCam, keypoints);
        cv::Mat allDescriptors(int(keypoints.size()), 48, CV_8UC1);
        for (size_t k = 0; k < keypoints.size(); ++k) {
          std::memcpy(allDescriptors.data + 48 * k, descriptors.at(k).data, 48);
        }
        multiFrame->resetDescriptors(curCam, allDescriptors);
        multiFrame->computeBackProjections(curCam);
        keypoints.clear();
        descriptors.clear();
      };
      while (sqlite3_step(q.get()) == SQLITE_ROW) {
        sqlite3_stmt *s = q.get();
        uint64_t stateId64 = uint64_t(sqlite3_column_int64(s, 0));
        uint64_t cameraIdx = uint64_t(sqlite3_column_int64(s, 1));
        if (!have || stateId64 != curState || cameraIdx != curCam) {
          flush();
          curState = stateId64;
          curCam = cameraIdx;
          have = true;
        }
        cv::KeyPoint kpt;
        kpt.pt.x = float(sqlite3_column_double(s, 3));
        kpt.pt.y = float(sqlite3_column_double(s, 4));
        kpt.size = float(sqlite3_column_double(s, 5));
        keypoints.push_back(kpt);
        const void *blob = sqlite3_column_blob(s, 6);
        OKVIS_ASSERT_TRUE(Exception, sqlite3_column_bytes(s, 6) == 48,
                          "descriptor blob size mismatch")
        cv::Mat descriptor(1, 48, CV_8UC1);
        std::memcpy(descriptor.data, blob, 48);
        descriptors.push_back(descriptor);
      }
      flush();
    }

    // -- 4. landmarks --
    {
      Stmt q(db, "SELECT landmark_id, x, y, z, quality FROM landmarks");
      while (sqlite3_step(q.get()) == SQLITE_ROW) {
        sqlite3_stmt *s = q.get();
        LandmarkId id(uint64_t(sqlite3_column_int64(s, 0)));
        Eigen::Vector4d r(sqlite3_column_double(s, 1), sqlite3_column_double(s, 2),
                          sqlite3_column_double(s, 3), 1.0);
        double quality = sqlite3_column_double(s, 4);
        fullGraph_->addLandmark(id, r, quality > 0.001);
        fullGraph_->setLandmarkQuality(id, quality);
      }
    }

    // -- 5. IMU edges + measurements --
    // Skipped when loading a prior map as a visual-only anchor: inertial
    // estimation (preintegration/bias) is always current-session only, so prior
    // IMU edges are never recreated in the live graph.
    if (loadImuEdges) {
      Stmt edges(db, "SELECT prev_state_id, state_id FROM imu_edges "
                     "ORDER BY prev_state_id, state_id");
      Stmt meas(db, "SELECT ts_ns, ax,ay,az, gx,gy,gz FROM imu_measurements "
                    "WHERE prev_state_id=? AND state_id=? ORDER BY ts_ns");
      while (sqlite3_step(edges.get()) == SQLITE_ROW) {
        uint64_t prev_state_id = uint64_t(sqlite3_column_int64(edges.get(), 0));
        uint64_t state_id = uint64_t(sqlite3_column_int64(edges.get(), 1));
        StateId prevStateId(prev_state_id);
        StateId stateId(state_id);

        ImuMeasurementDeque imuMeasurements;
        sqlite3_reset(meas.get());
        sqlite3_bind_int64(meas.get(), 1, sqlite3_int64(prev_state_id));
        sqlite3_bind_int64(meas.get(), 2, sqlite3_int64(state_id));
        while (sqlite3_step(meas.get()) == SQLITE_ROW) {
          sqlite3_stmt *m = meas.get();
          Time timestamp;
          timestamp.fromNSec(uint64_t(sqlite3_column_int64(m, 0)));
          Eigen::Vector3d acc(sqlite3_column_double(m, 1), sqlite3_column_double(m, 2),
                              sqlite3_column_double(m, 3));
          Eigen::Vector3d gyr(sqlite3_column_double(m, 4), sqlite3_column_double(m, 5),
                              sqlite3_column_double(m, 6));
          imuMeasurements.push_back(ImuMeasurement(timestamp, ImuSensorReadings(gyr, acc)));
        }

        OKVIS_ASSERT_TRUE(Exception, fullGraph_->states_.count(stateId),
                          "State " << state_id << " does not exist.");
        OKVIS_ASSERT_TRUE(Exception, fullGraph_->states_.count(prevStateId),
                          "State " << prev_state_id << " does not exist.");
        ViGraph::State &state0 = fullGraph_->states_.at(prevStateId);
        ViGraph::State &state1 = fullGraph_->states_.at(stateId);
        state0.nextImuLink.errorTerm.reset(new ceres::ImuError(
          imuMeasurements, imuParameters_, state0.timestamp, state1.timestamp));
        state1.previousImuLink.errorTerm = state0.nextImuLink.errorTerm;
        state0.nextImuLink.residualBlockId = fullGraph_->problem_->AddResidualBlock(
          state0.nextImuLink.errorTerm.get(), nullptr, state0.pose->parameters(),
          state0.speedAndBias->parameters(), state1.pose->parameters(),
          state1.speedAndBias->parameters());
        state1.previousImuLink.residualBlockId = state0.nextImuLink.residualBlockId;
      }
    }

    // -- 6. observations --
    {
      Stmt q(db, "SELECT state_id, cam_idx, kp_idx, landmark_id, u, v "
                 "FROM observations");
      while (sqlite3_step(q.get()) == SQLITE_ROW) {
        sqlite3_stmt *s = q.get();
        uint64_t stateId64 = uint64_t(sqlite3_column_int64(s, 0));
        int cameraIdx = sqlite3_column_int(s, 1);
        int keypointIdx = sqlite3_column_int(s, 2);
        uint64_t lmId64 = uint64_t(sqlite3_column_int64(s, 3));
        StateId stateId(stateId64);
        LandmarkId lmId(lmId64);

        OKVIS_ASSERT_TRUE(Exception, multiFrames_.count(stateId),
                          "Observation to non-existant multi-frame")
        MultiFramePtr multiFrame = multiFrames_.at(stateId);

        multiFrame->setLandmarkId(cameraIdx, keypointIdx, lmId64);
        KeypointIdentifier kid(stateId64, cameraIdx, keypointIdx);
        switch (distType) {
        case okvis::cameras::NCameraSystem::RadialTangential:
          fullGraph_->addObservation<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
            *multiFrame, lmId, kid, true);
          break;
        case okvis::cameras::NCameraSystem::Equidistant:
          fullGraph_->addObservation<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
            *multiFrame, lmId, kid, true);
          break;
        case okvis::cameras::NCameraSystem::RadialTangential8:
          fullGraph_->addObservation<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
            *multiFrame, lmId, kid, true);
          break;
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
        }

        // also store the 3D point
        Eigen::Vector4d hPoint_W = fullGraph_->landmark(lmId);
        kinematics::Transformation T_WS = fullGraph_->pose(stateId);
        Eigen::Vector4d hPoint_S = T_WS.inverse() * hPoint_W;
        multiFrame->setLandmark(cameraIdx, keypointIdx, hPoint_S,
                                fullGraph_->landmarks_.at(lmId).quality > 0.05);
      }
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error loading map " << path << ": " << e.what();
    sqlite3_close(db);
    return false;
  }

  sqlite3_close(db);

  LOG(INFO) << "--- loaded component: ---";
  LOG(INFO) << "no. states: " << fullGraph_->states_.size();
  LOG(INFO) << "no. landmarks: " << fullGraph_->landmarks_.size();

  return true;
}

bool Component::save(const std::string &path)
{
  // (re)create the database
  sqlite3 *db = nullptr;
  // remove any pre-existing file so we always write a fresh map
  std::remove(path.c_str());
  if (sqlite3_open_v2(path.c_str(), &db,
                      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
    LOG(ERROR) << "Could not open " << path << " for writing: " << sqlite3_errmsg(db);
    sqlite3_close(db);
    return false;
  }

  try {
    // bulk-insert tuning
    execSql(db, "PRAGMA journal_mode=OFF;");
    execSql(db, "PRAGMA synchronous=OFF;");

    // schema
    execSql(db,
      "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);"
      "CREATE TABLE states(state_id INTEGER PRIMARY KEY, ts_ns INTEGER,"
      " tx REAL,ty REAL,tz REAL, qx REAL,qy REAL,qz REAL,qw REAL,"
      " vx REAL,vy REAL,vz REAL, bgx REAL,bgy REAL,bgz REAL, bax REAL,bay REAL,baz REAL);"
      "CREATE TABLE extrinsics(state_id INTEGER, cam_idx INTEGER, extr_id INTEGER,"
      " tx REAL,ty REAL,tz REAL, qx REAL,qy REAL,qz REAL,qw REAL, ts_ns INTEGER,"
      " PRIMARY KEY(state_id, cam_idx));"
      "CREATE TABLE keypoints(state_id INTEGER, cam_idx INTEGER, kp_idx INTEGER,"
      " x REAL, y REAL, size REAL, descriptor BLOB, landmark_id INTEGER,"
      " PRIMARY KEY(state_id, cam_idx, kp_idx));"
      "CREATE TABLE landmarks(landmark_id INTEGER PRIMARY KEY, x REAL,y REAL,z REAL,"
      " quality REAL);"
      "CREATE TABLE observations(state_id INTEGER, cam_idx INTEGER, kp_idx INTEGER,"
      " landmark_id INTEGER, u REAL, v REAL,"
      " info00 REAL, info01 REAL, info10 REAL, info11 REAL);"
      "CREATE TABLE imu_edges(prev_state_id INTEGER, state_id INTEGER,"
      " PRIMARY KEY(prev_state_id, state_id));"
      "CREATE TABLE imu_measurements(prev_state_id INTEGER, state_id INTEGER, ts_ns INTEGER,"
      " ax REAL,ay REAL,az REAL, gx REAL,gy REAL,gz REAL);");

    execSql(db, "BEGIN TRANSACTION;");

    // meta
    {
      Stmt q(db, "INSERT INTO meta(key,value) VALUES(?,?)");
      auto put = [&](const std::string &k, const std::string &v) {
        sqlite3_bind_text(q.get(), 1, k.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(q.get(), 2, v.c_str(), -1, SQLITE_TRANSIENT);
        q.stepDone();
      };
      put("format_version", std::to_string(kMapFormatVersion));
      put("descriptor_type", "BRISK2");
      put("num_cameras", std::to_string(nCameraSystem_.numCameras()));
      put("distortion_type", std::to_string(int(nCameraSystem_.distortionType(0))));
    }

    Stmt stateStmt(db,
      "INSERT INTO states(state_id, ts_ns, tx,ty,tz, qx,qy,qz,qw,"
      " vx,vy,vz, bgx,bgy,bgz, bax,bay,baz)"
      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    Stmt extrStmt(db,
      "INSERT INTO extrinsics(state_id, cam_idx, extr_id, tx,ty,tz, qx,qy,qz,qw, ts_ns)"
      " VALUES(?,?,?,?,?,?,?,?,?,?,?)");
    Stmt kpStmt(db,
      "INSERT INTO keypoints(state_id, cam_idx, kp_idx, x, y, size, descriptor, landmark_id)"
      " VALUES(?,?,?,?,?,?,?,?)");
    Stmt lmStmt(db,
      "INSERT INTO landmarks(landmark_id, x,y,z, quality) VALUES(?,?,?,?,?)");
    Stmt obsStmt(db,
      "INSERT INTO observations(state_id, cam_idx, kp_idx, landmark_id, u, v,"
      " info00, info01, info10, info11) VALUES(?,?,?,?,?,?,?,?,?,?)");
    Stmt imuEdgeStmt(db,
      "INSERT INTO imu_edges(prev_state_id, state_id) VALUES(?,?)");
    Stmt imuMeasStmt(db,
      "INSERT INTO imu_measurements(prev_state_id, state_id, ts_ns, ax,ay,az, gx,gy,gz)"
      " VALUES(?,?,?,?,?,?,?,?,?)");

    std::set<LandmarkId> writtenLandmarks;  // only save landmarks once

    for (auto iter = fullGraph_->states_.begin(); iter != fullGraph_->states_.end(); ++iter) {
      const StateId id = iter->first;
      const ViGraph::State &state = iter->second;

      // state row (pose + speed/bias)
      const kinematics::Transformation &T_WS = state.pose->estimate();
      const SpeedAndBias &sb = state.speedAndBias->estimate();
      sqlite3_stmt *s = stateStmt.get();
      sqlite3_bind_int64(s, 1, sqlite3_int64(id.value()));
      sqlite3_bind_int64(s, 2, sqlite3_int64(state.timestamp.toNSec()));
      sqlite3_bind_double(s, 3, T_WS.r()[0]);
      sqlite3_bind_double(s, 4, T_WS.r()[1]);
      sqlite3_bind_double(s, 5, T_WS.r()[2]);
      sqlite3_bind_double(s, 6, T_WS.q().x());
      sqlite3_bind_double(s, 7, T_WS.q().y());
      sqlite3_bind_double(s, 8, T_WS.q().z());
      sqlite3_bind_double(s, 9, T_WS.q().w());
      sqlite3_bind_double(s, 10, sb[0]);
      sqlite3_bind_double(s, 11, sb[1]);
      sqlite3_bind_double(s, 12, sb[2]);
      sqlite3_bind_double(s, 13, sb[3]);
      sqlite3_bind_double(s, 14, sb[4]);
      sqlite3_bind_double(s, 15, sb[5]);
      sqlite3_bind_double(s, 16, sb[6]);
      sqlite3_bind_double(s, 17, sb[7]);
      sqlite3_bind_double(s, 18, sb[8]);
      stateStmt.stepDone();

      // frames / extrinsics + keypoints
      MultiFramePtr multiFrame = multiFrames_.at(id);
      for (uint64_t i = 0; i < multiFrame->numFrames(); ++i) {
        const kinematics::Transformation T_SC = state.extrinsics.at(i)->estimate();
        sqlite3_stmt *e = extrStmt.get();
        sqlite3_bind_int64(e, 1, sqlite3_int64(id.value()));
        sqlite3_bind_int64(e, 2, sqlite3_int64(i));
        sqlite3_bind_int64(e, 3, sqlite3_int64(state.extrinsics.at(i)->id()));
        sqlite3_bind_double(e, 4, T_SC.r()[0]);
        sqlite3_bind_double(e, 5, T_SC.r()[1]);
        sqlite3_bind_double(e, 6, T_SC.r()[2]);
        sqlite3_bind_double(e, 7, T_SC.q().x());
        sqlite3_bind_double(e, 8, T_SC.q().y());
        sqlite3_bind_double(e, 9, T_SC.q().z());
        sqlite3_bind_double(e, 10, T_SC.q().w());
        sqlite3_bind_int64(e, 11, sqlite3_int64(multiFrame->timestamp().toNSec()));
        extrStmt.stepDone();

        for (uint64_t k = 0; k < multiFrame->numKeypoints(i); ++k) {
          cv::KeyPoint cvKeypoint;
          multiFrame->getCvKeypoint(i, k, cvKeypoint);
          const unsigned char *descriptor = multiFrame->keypointDescriptor(i, k);
          sqlite3_stmt *kp = kpStmt.get();
          sqlite3_bind_int64(kp, 1, sqlite3_int64(id.value()));
          sqlite3_bind_int64(kp, 2, sqlite3_int64(i));
          sqlite3_bind_int64(kp, 3, sqlite3_int64(k));
          sqlite3_bind_double(kp, 4, cvKeypoint.pt.x);
          sqlite3_bind_double(kp, 5, cvKeypoint.pt.y);
          sqlite3_bind_double(kp, 6, cvKeypoint.size);
          sqlite3_bind_blob(kp, 7, descriptor, 48, SQLITE_TRANSIENT);
          sqlite3_bind_int64(kp, 8, sqlite3_int64(multiFrame->landmarkId(i, k)));
          kpStmt.stepDone();
        }
      }

      // IMU edge (the link to the previous state)
      if (state.previousImuLink.errorTerm) {
        const auto imuError =
          std::dynamic_pointer_cast<ceres::ImuError>(state.previousImuLink.errorTerm);
        if (imuError) {
          auto iterPrev = iter;
          --iterPrev;
          sqlite3_stmt *ie = imuEdgeStmt.get();
          sqlite3_bind_int64(ie, 1, sqlite3_int64(iterPrev->first.value()));
          sqlite3_bind_int64(ie, 2, sqlite3_int64(id.value()));
          imuEdgeStmt.stepDone();

          const okvis::ImuMeasurementDeque imuMeasurements = imuError->imuMeasurements();
          for (auto m = imuMeasurements.begin(); m != imuMeasurements.end(); ++m) {
            const Eigen::Vector3d acc = m->measurement.accelerometers;
            const Eigen::Vector3d gyr = m->measurement.gyroscopes;
            sqlite3_stmt *im = imuMeasStmt.get();
            sqlite3_bind_int64(im, 1, sqlite3_int64(iterPrev->first.value()));
            sqlite3_bind_int64(im, 2, sqlite3_int64(id.value()));
            sqlite3_bind_int64(im, 3, sqlite3_int64(m->timeStamp.toNSec()));
            sqlite3_bind_double(im, 4, acc[0]);
            sqlite3_bind_double(im, 5, acc[1]);
            sqlite3_bind_double(im, 6, acc[2]);
            sqlite3_bind_double(im, 7, gyr[0]);
            sqlite3_bind_double(im, 8, gyr[1]);
            sqlite3_bind_double(im, 9, gyr[2]);
            imuMeasStmt.stepDone();
          }
        }
      }

      // observations + landmarks
      for (const auto &obs : state.observations) {
        const LandmarkId lmId = obs.second.landmarkId;
        if (!writtenLandmarks.count(lmId)) {
          const ViGraph::Landmark &landmark = fullGraph_->landmarks_.at(lmId);
          const Eigen::Vector4d hPt = landmark.hPoint->estimate();
          sqlite3_stmt *lm = lmStmt.get();
          sqlite3_bind_int64(lm, 1, sqlite3_int64(lmId.value()));
          sqlite3_bind_double(lm, 2, hPt[0] / hPt[3]);
          sqlite3_bind_double(lm, 3, hPt[1] / hPt[3]);
          sqlite3_bind_double(lm, 4, hPt[2] / hPt[3]);
          sqlite3_bind_double(lm, 5, landmark.quality);
          lmStmt.stepDone();
          writtenLandmarks.insert(lmId);
        }

        const Eigen::Matrix2d info = obs.second.errorTerm->information();
        const Eigen::Vector2d coord = obs.second.errorTerm->measurement();
        sqlite3_stmt *o = obsStmt.get();
        sqlite3_bind_int64(o, 1, sqlite3_int64(id.value()));
        sqlite3_bind_int64(o, 2, sqlite3_int64(obs.first.cameraIndex));
        sqlite3_bind_int64(o, 3, sqlite3_int64(obs.first.keypointIndex));
        sqlite3_bind_int64(o, 4, sqlite3_int64(lmId.value()));
        sqlite3_bind_double(o, 5, coord[0]);
        sqlite3_bind_double(o, 6, coord[1]);
        sqlite3_bind_double(o, 7, info(0, 0));
        sqlite3_bind_double(o, 8, info(0, 1));
        sqlite3_bind_double(o, 9, info(1, 0));
        sqlite3_bind_double(o, 10, info(1, 1));
        obsStmt.stepDone();
      }
    }

    execSql(db, "COMMIT;");

    // indexes for fast reload
    execSql(db, "CREATE INDEX idx_obs_state ON observations(state_id);");
    execSql(db, "CREATE INDEX idx_kp_state ON keypoints(state_id);");
    execSql(db, "CREATE INDEX idx_imumeas ON imu_measurements(prev_state_id, state_id);");
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error saving map " << path << ": " << e.what();
    sqlite3_close(db);
    return false;
  }

  sqlite3_close(db);
  return true;
}

}  // namespace okvis
