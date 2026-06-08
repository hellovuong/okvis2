# Map IO, Relocalization & Continuous Mapping — Design & Implementation

**Project:** OKVIS2 (Open Keyframe-based Visual-Inertial SLAM)
**Feature:** Persist the SLAM map to disk, reload it, relocalize a new live
session against it, and continue mapping (multi-session).
**On-disk format:** a single **SQLite3** database holding the full,
re-optimisable visual-inertial graph.
**Status:** Implemented and **validated end-to-end on TUM-VI 512 (room1 → room2)**
— save, reload, relocalize, continuous mapping, and coherent combined re-save all
work with no crashes (see [§11 Validation status](#11-validation-status)).

> ### ⚠️ Architecture revision (v2 — the one that is implemented)
> The continuous-mapping design below was originally a *"frozen anchor"* that
> loaded the prior map's **IMU edges** into the live graph (single inertial graph
> across sessions). That **crashed** on real data because prior-session frames
> (with prior-session timestamps) leaked into the live inertial/state path. The
> **implemented** architecture is **visual-only lifelong mapping**:
> - The prior map contributes **only visual factors** (landmarks + keyframe
>   observations). **IMU is strictly current-session** — preintegration,
>   propagation, prediction and bias estimation only ever touch live frames.
> - The previous session's **final bias is reused as the live init bias**.
> - Prior **poses/extrinsics/speed-bias are frozen** (stable gauge); prior
>   **landmarks stay variable** so live observations refine them across runs
>   (lifelong — the map does not go out of date).
> - Relocalization (`alignToPriorMap`) does a **one-shot rigid alignment** to
>   bootstrap projection, then relies on the reprojection observations
>   `matchToMap` adds to the prior landmarks — **no** cross-session pose-graph
>   constraint and **no** full-graph-optimisation trigger.
>
> Read §7 and §11 with this revision in mind; the per-file log in §10 and the
> validation in §11 reflect the implemented (v2) code.

---

## Table of contents

1. [Goals & scope](#1-goals--scope)
2. [Starting point: what already existed](#2-starting-point-what-already-existed)
3. [Key design decisions (and the alternatives rejected)](#3-key-design-decisions-and-the-alternatives-rejected)
4. [Solution architecture](#4-solution-architecture)
5. [Layer A — SQLite map format](#5-layer-a--sqlite-map-format)
6. [Layer B — Load at startup & relocalization](#6-layer-b--load-at-startup--relocalization)
7. [Layer C — Continuous mapping (frozen anchor)](#7-layer-c--continuous-mapping-frozen-anchor)
8. [The `rbegin()` / ID-ordering constraint (deep dive)](#8-the-rbegin--id-ordering-constraint-deep-dive)
9. [Build-system changes & the Ceres fix](#9-build-system-changes--the-ceres-fix)
10. [File-by-file change log](#10-file-by-file-change-log)
11. [Validation status](#11-validation-status)
12. [Known risks & limitations](#12-known-risks--limitations)
13. [How to use it](#13-how-to-use-it)
14. [Future work](#14-future-work)

---

## 1. Goals & scope

The system previously produced a map only as a side effect of a run, wrote it
only at shutdown, and in a verbose ASCII format that nothing reloaded in
practice. The goal of this work was three capabilities, all confirmed with the
product owner up front:

1. **Save & reload a map** — persist a complete map and load it back.
2. **Relocalization** — start a fresh live session and recover the camera pose
   relative to a previously-saved map.
3. **Continuous mapping** — having relocalized, keep mapping in the prior map's
   coordinate frame and write the (extended) map back out as one coherent map.

The persistence format was chosen to be **SQLite3** (the user's request),
storing the **full re-optimisable graph** (not just a lightweight localization
map), and the format **replaces** the old text/CSV format rather than living
alongside it.

---

## 2. Starting point: what already existed

A thorough read of the codebase (three parallel exploration passes) found that a
*partial, disconnected* multi-session scaffold already existed but was never
wired up end-to-end:

| Piece | Location | State |
|---|---|---|
| `Component::save` / `load` | `okvis_ceres/src/Component.cpp` | Worked. Serialised the **full graph** to a custom ASCII "g2o-like" text format: poses (`VERTEX_SE3:QUAT_TIME`), velocity/biases (`VERTEX_R3:*`), landmarks (`VERTEX_TRACKXYZ`), frames + keypoints + BRISK2 descriptors (`FRAME`/`FRAME:KEYPOINT`), IMU edges with raw measurements (`EDGE_IMU`/`EDGE_IMU:MEASUREMENTS`), observations (`EDGE_OBS`). |
| `ViSlamBackend::saveMap` | `okvis_ceres/src/ViSlamBackend.cpp` | Worked. Wrapped `Component::save` (`.g2o`) **plus** a second custom `.csv` with landmarks + per-frame covisibilities + descriptors. Wired into apps at shutdown via `setMapCsvFile`/`saveMap`. |
| `Frontend::loadComponent` | `okvis_frontend/src/Frontend.cpp` | Worked but **never called** (commented out in the `ThreadedSlam` constructor). Loaded a `Component` into a *separate* graph and built a per-component DBoW place-recognition database (`components_`, `componentDBows_`, `componentsFixed_`). |
| Relocalization-against-component | `Frontend::dataAssociationAndInitialization`, ~L804–847 | Partially worked. Queried the component DBoW, geometrically verified with `verifyRecognisedPlace` (3D–2D RANSAC + refine), and computed an alignment transform — **but stored it in `ViSlamBackend::T_AiS_`, which was written and never read.** So relocalization computed an answer that nothing consumed. |
| Loop closure | `ViSlamBackend::attemptLoopClosure`, `addLoopClosureFrame` | Worked for *intra-session* loops. Distributes drift along one continuous trajectory and merges landmarks. |

### Canonical map state

The map lives in `ViSlamBackend`:

- `realtimeGraph_` and `fullGraph_` — two `ViGraphEstimator` instances (active
  window + asynchronous background graph). Each owns a `ceres::Problem` plus
  `states_` (`map<StateId,State>`), `landmarks_` (`map<LandmarkId,Landmark>`),
  and `observations_`.
- `multiFrames_` (`map<StateId,MultiFramePtr>`) — images, keypoints, BRISK2
  descriptors, per-keypoint landmark associations.
- `auxiliaryStates_` (`map<StateId,AuxiliaryState>`) — per-state flags:
  `isKeyframe`, `isImuFrame`, `isPoseGraphFrame`, `isPlaceRecognitionFrame`,
  `loopId`, `closedLoop`, …
- Membership sets `keyFrames_`, `imuFrames_`, `loopClosureFrames_`.

### ID generation (critical)

`StateId` and `LandmarkId` are sequential `uint64` starting at 1, generated as
**`max-existing-id + 1`** via `states_.rbegin()->first + 1`
(`ViGraph.cpp`) and `landmarks_.rbegin()->first + 1`. There is no global counter.
This fact drives the whole merge design (see [§8](#8-the-rbegin--id-ordering-constraint-deep-dive)).

---

## 3. Key design decisions (and the alternatives rejected)

Four decisions shaped the implementation. Each was made explicitly (three with
the product owner) after surfacing the trade-offs.

### Decision 1 — SQLite *replaces* the text format

**Chosen:** Migrate `Component::save`/`load` to SQLite as the primary, canonical,
re-loadable format; keep only the human-readable trajectory CSV for debugging.

Rejected: adding SQLite as a *parallel* backend (more surface, two formats to
keep in sync) or keeping CSV map export (nothing read it back).

### Decision 2 — Store the *full re-optimisable graph*

**Chosen:** Persist everything needed to re-run bundle adjustment: poses,
speed/bias, landmarks, observations, descriptors, **and** the raw IMU
measurements per edge.

Rejected: a lightweight "localization map" (keyframes + landmarks + descriptors
only). The full graph is required for true continuous mapping / re-optimization
and matches what `Component` already serialised.

### Decision 3 — Merge strategy: *startup-load single graph*

This was the pivotal architectural decision. Three strategies were presented:

| Strategy | Idea | Verdict |
|---|---|---|
| **(a) Startup-load single graph** ✅ | Load the prior map into the live graph **at startup** (prior ids `1..N`, live ids continue at `N+1`). One graph from the start. | **Chosen.** Lowest architectural risk: `rbegin()` stays valid (live ids always highest), no ID collisions, cross-session linking reuses existing machinery. |
| (b) Merge-at-save | Keep prior map separate at runtime; align the live trajectory and write both into one DB at save time. | Rejected. No true joint optimization; "co-optimization" limited to one constraint. |
| (c) Runtime merge (original plan) | Inject prior frames into the live graph *during* a runtime relocalization, with explicit ID counters + reworking ~20 `rbegin()` sites. | Rejected. Largest, riskiest change; fights the ID/`rbegin()` architecture badly. |

The decisive insight: because the prior map is loaded with ids `1..N` and the
live session starts at `N+1`, `states_.rbegin()` (used pervasively to mean *"the
newest live frame in time"*) keeps pointing at the newest live frame, and the
two sessions are simply **disconnected components in one graph**, joined later by
a relocalization constraint. This avoids a renumbering pass and avoids poisoning
the ~20 `rbegin()` call sites.

### Decision 4 — Prior map is a *frozen anchor*

Three policies for how prior frames behave in the optimizer/marginalizer were
presented:

| Policy | Behaviour | Verdict |
|---|---|---|
| **Frozen anchor** ✅ | Prior poses & landmarks are `SetParameterBlockConstant` and never marginalised. Live localises against them and adds **new** landmarks. | **Chosen.** Simplest, lowest-risk, reproducible. The prior map stays pristine; memory grows with prior size (kept resident — accepted). |
| Frozen + pose-graph | Prior values fixed but converted to pose-graph frames outside the active window (bounded memory). | Rejected for v1 (more core integration). |
| Fully co-optimized | Prior frames free to move/marginalise; joint BA refines both. | Rejected (prior map can drift; highest risk). |

A consequence of "frozen anchor" that drove implementation: prior frames are
deliberately kept **out of `keyFrames_`/`imuFrames_`** (in a dedicated
`priorFrames_` set), so the marginalisation strategy never selects them for
elimination, while `matchToMap` still sees their (resident) landmarks.

---

## 4. Solution architecture

The work is organised in three layers, landed in order:

```
Layer A  SQLite persistence            Component::save / loadInto  (full graph <-> .db)
Layer B  Load-at-startup + relocalize  ThreadedSlam::loadMap, frontend routing
Layer C  Continuous mapping            frozen-anchor prior in one graph + alignToPriorMap
```

End-to-end data flow:

```
 SESSION 1 (mapping)
   run SLAM ──► fullGraph_ + multiFrames_ ──► ViSlamBackend::saveMap()
                                              └─► Component::save() ──► map.db

 SESSION 2 (relocalize + continue)
   --load-map map.db
     └─► ThreadedSlam::loadMap()
           ├─► ViSlamBackend::loadMap()
           │     ├─ Component::loadInto(map.db, realtimeGraph_, multiFrames_)
           │     ├─ Component::loadInto(map.db, fullGraph_,     multiFrames_)
           │     ├─ register prior frames in priorFrames_ / auxiliaryStates_ / anyState_
           │     └─ freeze prior poses + speed/bias + extrinsics + landmarks (constant)
           └─► Frontend::addPriorMapFrames()  → prior frames into the place-recognition DBoW

   live frame 1 ──► addStates() initialises a fresh live world at id = N+1
   live frame k ──► matchToMap (prior + live landmarks) + DBoW place recognition
                     └─ DBoW match to a prior frame? ──► alignToPriorMap()
                          ├─ rigidly map the live world onto the prior world
                          ├─ add a relative-pose constraint (prior ↔ live)
                          └─ trigger full-graph optimisation
   subsequent frames ──► matchToMap now hits frozen prior landmarks directly → continuous localization
   shutdown ──► saveMap() writes ONE coherent .db (prior ids 1..N + live ids N+1..)
```

---

## 5. Layer A — SQLite map format

### 5.1 Where it lives

`Component` already owned both the graph and the multiframes and knew how to walk
them (`save`) and rebuild them (`load`). The public API (`save(path)`,
`load(path)`) is unchanged; only the I/O was swapped from text streams to SQLite.
`okvis_ceres` now links `SQLite::SQLite3` (CMake's `FindSQLite3`).

A small RAII layer wraps the C API inside `Component.cpp`:

- `execSql(db, sql)` — run a statement, throw `Component::Exception` on error.
- `class Stmt` — RAII `sqlite3_stmt` with `stepDone()` for INSERT/UPDATE
  (steps to `SQLITE_DONE`, then `sqlite3_reset`).

### 5.2 Schema (one file = one full graph)

```sql
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
  -- format_version, descriptor_type ("BRISK2"), num_cameras, distortion_type

CREATE TABLE states(state_id INTEGER PRIMARY KEY, ts_ns INTEGER,
  tx,ty,tz, qx,qy,qz,qw,                 -- T_WS pose
  vx,vy,vz, bgx,bgy,bgz, bax,bay,baz);   -- speed/bias  [v(0-2), gyrbias(3-5), accbias(6-8)]

CREATE TABLE extrinsics(state_id, cam_idx, extr_id,
  tx,ty,tz, qx,qy,qz,qw, ts_ns, PRIMARY KEY(state_id,cam_idx));

CREATE TABLE keypoints(state_id, cam_idx, kp_idx,
  x, y, size, descriptor BLOB, landmark_id, PRIMARY KEY(state_id,cam_idx,kp_idx));
  -- descriptor is a 48-byte BRISK2 BLOB (was 96 hex chars in the text format)

CREATE TABLE landmarks(landmark_id INTEGER PRIMARY KEY, x,y,z, quality);

CREATE TABLE observations(state_id, cam_idx, kp_idx, landmark_id,
  u, v, info00, info01, info10, info11);

CREATE TABLE imu_edges(prev_state_id, state_id, PRIMARY KEY(prev_state_id,state_id));
CREATE TABLE imu_measurements(prev_state_id, state_id, ts_ns, ax,ay,az, gx,gy,gz);
```

The schema maps **one-to-one** onto the old text records, so no information is
lost in the migration. Write path tuning: `PRAGMA journal_mode=OFF`,
`synchronous=OFF`, a single `BEGIN…COMMIT` transaction, prepared statements, and
post-insert indexes on `observations(state_id)`, `keypoints(state_id)`,
`imu_measurements(prev_state_id,state_id)`.

### 5.3 Save (`Component::save`)

Iterates `fullGraph_->states_` in id order. For each state writes the `states`
row (pose + speed/bias), the per-camera `extrinsics` and all `keypoints`
(descriptor as a BLOB), the `imu_edges`/`imu_measurements` for the link to the
previous state (only for `ceres::ImuError` links), and the `observations` + the
referenced `landmarks` (deduplicated via a `writtenLandmarks` set — same as the
original). The file is removed first so every save is fresh.

### 5.4 Load (`Component::load` → `Component::loadInto`)

`load(path)` now allocates an owned `ViGraphEstimator` and delegates to a new,
reusable:

```cpp
bool Component::loadInto(const std::string &path,
                         ViGraphEstimator &graph,
                         std::map<StateId, MultiFramePtr> &frames);
```

`loadInto` carries the **exact parameter-block-creation order** from the original
loader (this ordering is load-bearing and was preserved deliberately):

1. **states** — create `PoseParameterBlock` (+ `poseManifold_`) and
   `SpeedAndBiasParameterBlock`, add to the ceres problem, set timestamp.
2. **extrinsics / frames** — create/reuse the `MultiFrame`; create per-`extr_id`
   `PoseParameterBlock`s (shared across cameras/states with the same id).
3. **keypoints** — grouped by `(state_id, cam_idx)`; `resetKeypoints`,
   `resetDescriptors`, `computeBackProjections` per camera.
4. **landmarks** — `addLandmark(id, r, quality>0.001)` + `setLandmarkQuality`.
5. **IMU edges** — rebuild `ceres::ImuError` from the raw measurements and add
   the residual block between the two states.
6. **observations** — `setLandmarkId`, `addObservation<GEOMETRY>` (templated on
   distortion type), and store the back-projected 3D point in the multiframe.

Why `loadInto` exists: two `ceres::Problem`s cannot be merged, so to put the
prior map into **both** `realtimeGraph_` and `fullGraph_`, the backend calls
`loadInto` twice with the **same** `frames` map. The first call creates the
multiframes; the second reuses them (sharing `MultiFramePtr`) while building its
own parameter/residual blocks in the second graph's problem.

> **Bug found & fixed during review.** The original extrinsics resize was gated on
> *"is this a newly-created multiframe"* (`firstFrame`). On the second
> `loadInto` call the multiframe already exists, so the second graph's
> `state.extrinsics` would never be resized and `state.extrinsics.at(cameraIdx)`
> would throw. Fixed to resize based on the **per-graph** state:
> `if (state.extrinsics.size() != multiFrame->numFrames()) state.extrinsics.resize(...)`.

### 5.5 Behaviour preserved from the text loader

As in the original text loader, `addObservation` **recomputes** the 2D
measurement and information matrix from the (loaded) keypoint rather than from
the stored `u,v`/`info` columns. Those columns are persisted for completeness/
debugging but intentionally not consumed on load — this matches the pre-existing
behaviour exactly (the old `EDGE_OBS` parser also parsed and then ignored them).

### 5.6 Caller wiring

`ViSlamBackend::saveMap` was reduced from "write `.g2o` + custom `.csv`" to:
normalise the path to `*.db`, build a `Component` from `fullGraph_` +
`multiFrames_`, and call `component.save(dbPath)`. The synchronous app and the
ROS2 node now pass `…-final_map.db`. The human-readable trajectory CSV
(`writeFinalTrajectoryCsv`) is untouched.

---

## 6. Layer B — Load at startup & relocalization

### 6.1 Entry point

```cpp
bool ThreadedSlam::loadMap(const std::string & path);   // call once, before any frames
```

It calls `estimator_.loadMap(path, parameters_.nCameraSystem, parameters_.imu, priorFrames)`
then `frontend_.addPriorMapFrames(priorFrames)`. It must run before any
images/IMU are processed; at that point the optimisation threads are idle
(blocking on empty input queues), so the estimator graphs are quiescent.

The synchronous app exposes it as a CLI flag:

```
./okvis_app_synchronous <config.yaml> <dataset> [-rpg] [--load-map <prior-map.db>]
```

(Argument parsing was relaxed from "exactly 3 or 4 args" to a small scan so the
flag can appear after the positional args.)

### 6.2 Consuming the relocalization result

The previously-dead `T_AiS_` path is replaced by a live one. Because the prior
frames are loaded into the **main** graph and registered in the **main**
place-recognition database (`dBow_`), relocalization flows through the existing
loop-closure query in `Frontend::dataAssociationAndInitialization`. A new branch
runs **before** the intra-session loop-closure gate:

```cpp
// (inside the DBoW candidate loop, after fetching the matched old frame)
if (estimator.isPriorFrame(id.first)) {
  if (estimator.isPriorMapAligned()) continue;          // align once
  if (!verifyRecognisedPlace(estimator, params, framesInOut, oldFrame, T_Sold_Snew, H, 10)) {
    attempts++; continue;                                // 3D-2D RANSAC + refine
  }
  attempts++;
  if (estimator.alignToPriorMap(id.first, StateId(framesInOut->id()), T_Sold_Snew, H)) {
    matchToMap<…>(estimator, params, framesInOut->id()); // match against now-aligned prior landmarks
    *asKeyframe = doWeNeedANewKeyframe(estimator, framesInOut);
  }
  break;
}
// … existing intra-session loop-closure path (requires isPoseGraphFrame) …
```

The branch must precede the `isPoseGraphFrame(id.first)` gate, because prior
frames are **not** pose-graph frames and would otherwise be filtered out.

`addPriorMapFrames` mirrors the DBoW-fill logic from `loadComponent`: per-keypoint
48-byte BRISK descriptors are assembled across cameras and added to
`dBow_->database`, with the prior `StateId` pushed onto `dBow_->poseIds`.

---

## 7. Layer C — Continuous mapping (visual-only lifelong anchor)

> See the **v2 revision note** at the top: the implemented design loads the prior
> as a **visual-only** anchor (no cross-session IMU), keeps prior **landmarks
> variable** (lifelong), seeds the live init **bias** from the prior session, and
> `alignToPriorMap` adds **no** pose-graph constraint / full-graph-opt trigger. The
> subsections below describe the original frozen-anchor reasoning; the code matches
> v2.

### 7.1 `ViSlamBackend::loadMap`

```cpp
bool loadMap(const std::string & path,
             const cameras::NCameraSystem & nCameraSystem,
             const ImuParameters & imuParameters,
             std::map<StateId, MultiFramePtr> & priorFramesOut);
```

1. Assert no live state exists yet.
2. `loadInto` the DB into `realtimeGraph_` and then `fullGraph_` (shared frames).
3. Record `priorMaxStateId_` and `priorMaxLandmarkId_`.
4. For every prior state: insert into `priorFrames_`; create an `AuxiliaryState`
   with `isKeyframe=true`, `isImuFrame=false`, `isPoseGraphFrame=false`,
   `isPlaceRecognitionFrame=true`, `loopId=id`; add an `anyState_` entry (mirrors
   `addStatesInitialise` for trajectory interpolation); export the multiframe to
   `priorFramesOut`. **They are deliberately not added to `keyFrames_`/`imuFrames_`.**
5. **Freeze** the prior region in both graphs:
   `freezePosesUntil(priorMaxStateId_)`, `freezeSpeedAndBiasesUntil(...)`,
   `freezeExtrinsicsUntil(...)`, and `SetParameterBlockConstant` on every prior
   landmark's parameter block. (`freezePosesUntil` walks from the given id down to
   the start, which is exactly the prior range since prior ids are the lowest.)

### 7.2 First live frame initialises above the prior

`addStates` decides initialise-vs-propagate. Previously it checked
`multiFrames_.empty()`; that is false once a prior map is loaded. It now checks
whether a **live** state exists:

```cpp
const bool liveUninitialised = realtimeGraph_.states_.empty()
    || realtimeGraph_.states_.rbegin()->first.value() <= priorMaxStateId_.value();
```

When uninitialised, it calls `addStatesInitialise`, which was generalised so that
a non-empty graph yields `id = rbegin()+1` (= `priorMax+1`) instead of asserting
empty and forcing `id=1`:

```cpp
StateId id = states_.empty() ? StateId(1) : StateId(states_.rbegin()->first + 1);
```

The first live frame thus gets a fresh, gravity-aligned **own** world frame at
`id = N+1`; it is *not* IMU-propagated from a prior frame. `addStates`'s return
was changed from `(id.value()==1 && id.isInitialised())` to `id.isInitialised()`
so a live-above-prior first frame reports success.

### 7.3 `ViSlamBackend::alignToPriorMap`

```cpp
bool alignToPriorMap(StateId priorFrame, StateId liveFrame,
                     const kinematics::Transformation & T_Sprior_Slive,
                     const Eigen::Matrix<double,6,6> & information);
```

On the first successful relocalization it rigidly snaps the live world onto the
prior world:

```
T_Wp_Slive = T_Wp_Sprior · T_Sprior_Slive          // desired live pose in prior world
T_Wp_Wl    = T_Wp_Slive  · T_Wl_Slive⁻¹             // live-world → prior-world rigid map
```

- Every **live** state (`id > priorMaxStateId_`) pose is left-multiplied by
  `T_Wp_Wl`; its velocity is rotated by `T_Wp_Wl.C()`. Prior states are skipped.
- Every **live** landmark (`id > priorMaxLandmarkId_`) is transformed by
  `T_Wp_Wl`. Prior landmarks are skipped (they stay the fixed reference).
- A relative-pose constraint `RelPoseInfo{T_Sprior_Slive, information,
  priorFrame, liveFrame}` is pushed onto `fullGraphRelativePoseConstraints_`
  (the same queue `attemptLoopClosure` uses; consumed at the start of a
  full-graph optimisation), `auxiliaryStates_.at(liveFrame).closedLoop = true`,
  `priorMapAligned_ = true`, and `needsFullGraphOptimisation_ = true`.

After alignment the live trajectory lives in the prior world frame, so the
ordinary `matchToMap` (which already iterates **all** resident landmarks,
including the frozen prior ones) connects live keypoints to prior landmarks every
frame — that is what makes mapping "continuous" without any further special case.

### 7.4 Why this reuses the loop-closure machinery cleanly

`attemptLoopClosure` distributes drift by walking `states_` from `pose_i` to the
end — correct when both frames are live (a live `pose_i` only iterates the live
region, since live ids are highest) but **wrong** for the first prior↔live
connection (it would walk through and corrupt the frozen prior frames). So the
prior↔live connection uses the dedicated rigid `alignToPriorMap`, while all
subsequent intra-live loop closures continue to use the unmodified
`attemptLoopClosure`.

### 7.5 Coherent re-save

Because prior + live states/landmarks/observations/multiframes all live in
`fullGraph_`/`multiFrames_` with disjoint ids, a second `saveMap` writes one
coherent `.db` containing both sessions, itself reloadable and re-optimisable.

---

## 8. The `rbegin()` / ID-ordering constraint (deep dive)

This constraint is why the architecture looks the way it does.

- IDs are generated `rbegin()+1`, i.e. "max + 1". ~20 sites in `ViSlamBackend`
  use `states_.rbegin()` to mean **"the newest live frame in time"** — IMU
  propagation base, pose freezing windows, marginalisation candidate windows,
  visualisation, trajectory output, `currentStateId`, etc.
- The original plan (runtime merge) would have inserted prior frames into the
  live graph **with ids above the live frames** (or forced a renumber), poisoning
  every one of those `rbegin()` assumptions — a large, high-risk change.
- The chosen "startup-load" design sidesteps this entirely: prior ids are
  `1..N` (the **lowest**), live ids start at `N+1` and only grow, so
  `states_.rbegin()` is always the newest live frame. The only adjustment needed
  was the first-live-frame init path (§7.2). Marginalisation is kept off the
  prior frames by not putting them in `keyFrames_`/`imuFrames_`.

The freeze/marginalisation code that walks backwards into the prior region (e.g.
`applyStrategy`'s freeze step) degrades gracefully: `freezePosesUntil` stops at
the first already-fixed pose, and prior poses are already fixed, so re-freezing
is idempotent. This was reasoned through, not yet run on data (see §12).

---

## 9. Build-system changes & the Ceres fix

### 9.1 SQLite

`okvis_ceres/CMakeLists.txt`: `find_package(SQLite3 REQUIRED)` and
`SQLite::SQLite3` added to the private link libraries. (CMake ships
`FindSQLite3`; system SQLite 3.45 was used.)

### 9.2 Top-level `Ceres::ceres` fix

A pre-existing latent bug surfaced once ROS was sourced (`BUILD_ROS2=ON`): the
`okvis_ros2` library is defined **directly in the top-level `CMakeLists.txt`**
and links `Ceres::ceres`, but `find_package(Ceres)` was only called inside the
`okvis_ceres`/`okvis_frontend` **subdirectory** scopes. Config-mode imported
targets created in a child scope do not propagate up to the parent scope, so the
top-level target failed with *"Target links to Ceres::ceres but the target was
not found."* Fix: add `find_package(Ceres REQUIRED)` at the **top level** (under
`if(USE_SYSTEM_CERES)`, right after `add_subdirectory(external)`), so the
imported target exists for every target in the tree.

---

## 10. File-by-file change log

`git diff --numstat` (excluding submodule pointer noise):

| File | +/− | What changed |
|---|---|---|
| `CMakeLists.txt` | +11/−1 | Top-level `find_package(Ceres)` fix (§9.2). |
| `okvis_ceres/CMakeLists.txt` | +2 | `find_package(SQLite3)` + link `SQLite::SQLite3`. |
| `okvis_ceres/include/okvis/Component.hpp` | +11/−1 | Declare `loadInto(path, graph, frames)`. |
| `okvis_ceres/src/Component.cpp` | +506/−398 | **Full rewrite** of `save`/`load` to SQLite; `load` delegates to new `loadInto`; RAII `Stmt`/`execSql`; schema; extrinsics-resize fix. |
| `okvis_ceres/include/okvis/ViSlamBackend.hpp` | +48/−2 | Declare `loadMap`, `alignToPriorMap`, `hasPriorMap`, `priorFrames`, `isPriorFrame`, `isPriorMapAligned`; add members `priorFrames_`, `priorMaxStateId_`, `priorMaxLandmarkId_`, `priorMapAligned_`. |
| `okvis_ceres/src/ViSlamBackend.cpp` | +159/−62 | `saveMap` → SQLite via `Component`; `addStates` init-vs-propagate uses `priorMaxStateId_` and returns `id.isInitialised()`; new `loadMap` + `alignToPriorMap`; reset of new members in `clear()`. |
| `okvis_ceres/src/ViGraph.cpp` | +5/−2 | `addStatesInitialise` generalised to non-empty graph (`id = rbegin()+1`). |
| `okvis_frontend/include/okvis/Frontend.hpp` | +9 | Declare `addPriorMapFrames`. |
| `okvis_frontend/src/Frontend.cpp` | +63 | Implement `addPriorMapFrames`; prior-frame relocalization branch in `dataAssociationAndInitialization`. |
| `okvis_multisensor_processing/include/okvis/ThreadedSlam.hpp` | +8/−1 | Declare `loadMap`. |
| `okvis_multisensor_processing/src/ThreadedSlam.cpp` | +15 | Implement `loadMap` (backend load + frontend DBoW registration). |
| `okvis_apps/src/okvis_app_synchronous.cpp` | +17/−5 | `--load-map` CLI flag + call; `.db` map path. |
| `okvis_ros2/src/okvis_node_synchronous.cpp` | +1/−1 | `.db` map path. |

---

## 11. Validation status

**Compilation:** builds cleanly via the canonical colcon command (Ninja + mold +
ccache, `RelWithDebInfo`, `NR_ENABLE_ASSERTS=ON`), exit 0.

**Behavioural — VALIDATED end-to-end on TUM-VI 512 (room1 → room2).** room1 and
room2 are different trajectories in the same physical room (overlapping appearance,
different absolute timestamps), which is the proper multi-session test.

Verification config: a copy of `config/tumvi_slam_512.yaml` with `use_cnn:false`
(faster, fewer deps) and `do_final_ba:true` (so the app actually calls `saveMap`).
Dataset path passed is the **`mav0`** directory; binary run from `build/okvis/`
(DBoW vocab + CNN model sit beside it).

| Test | Result |
|---|---|
| **1. Save (Layer A)** | room1 → `okvis2-slam-final_map.db` (14 MB): 258 states, 4253 landmarks, 48486 observations, 257 IMU edges, 75579 keypoints with 48-byte BRISK2 descriptor BLOBs. ✅ |
| **2. Reload** | `Component::loadInto` loaded 258 frames + 4253 landmarks into both graphs. ✅ |
| **3. Relocalization** | live room2 frame relocalized onto room1 prior frame 3 via DBoW + `verifyRecognisedPlace` RANSAC. ✅ |
| **4. Continuous mapping** | full room2 (2882 frames) + final BA, no crash; +263 live keyframes, +1571 new landmarks, prior landmarks refined, live→prior reprojection observations accumulated; bias seeded from prior session. ✅ |
| **5. Combined re-save** | one coherent `.db` (26 MB): 521 states (258 prior ids ≤2820 **+** 263 live ids >2820), 4867 landmarks (3296 prior + 1571 new), 97739 observations, **262 IMU edges — live-only** (confirms no cross-session IMU). ✅ |

Four crash bugs were found and fixed on real data while implementing the
visual-only (v2) architecture — all at the prior↔live boundary:

1. **Cross-session IMU** loaded into the live graph → `loadInto(loadImuEdges=false)`.
2. `currentStateId()` / `lastOptimisedState_` resolving to a **prior frame** (its
   timestamp fed the live IMU propagation → `imuMeasurements.front() <= time`
   assert). Fix: `currentStateId()` returns the newest *live* frame, uninitialised
   while only a prior map is loaded; `lastOptimisedState_` guarded on it.
3. The IMU-deque pop used the uninitialised `lastOptimisedState_.timestamp`
   (`Time(0) − overlap` → "Time is out of dual 32-bit range"). Fix: guard the pop
   on `lastOptimisedState_.id.isInitialised()`.
4. The publish loops did `imuMeasurementsByFrame_.at(priorId)` (only live frames
   are in that map) after a live loop closure refined the now-variable prior
   landmarks pulled prior frames into the updated set → `std::out_of_range:
   map::at`. Fix: skip prior frames in publication (they are the static visual
   anchor, not live trajectory).

An earlier 2-angle code review's findings were confirmed to be preserved-behavior
(observation info recomputation, §5.5) and graceful degradation (§8); one genuine
dev bug (second-graph extrinsics resize, §5.4) was caught there.

### Remaining tests worth running

- A live sequence that **starts away from** the prior map and relocalizes later
  (room2 happened to start overlapping room1's start, so relocalization fired on
  the first frame). This would exercise the drift-then-snap path.
- Reload the **combined** `.db` as a prior for a third session (multi-generation
  lifelong mapping) and watch landmark growth/pruning.
- EuRoC regression with no `--load-map` (output is now `.db`).

---

## 12. Known risks & limitations

- **Prior frames are resident but not in `keyFrames_`/`imuFrames_`.** This is the
  mechanism that keeps them frozen and un-marginalised, but any code that assumes
  "every state belongs to one of those membership sets" could mishandle them. The
  freeze/marginalisation paths reason out as safe (idempotent re-freeze), but this
  is unverified on data.
- **Memory grows with prior-map size** — the whole prior graph (poses, landmarks,
  observations, descriptors) stays resident, and all prior reprojection residuals
  remain in the ceres problem as constant-parameter blocks (correct but wasteful).
  This was the accepted trade-off for the "frozen anchor" policy.
- **Single prior map, single session join.** `alignToPriorMap` aligns once
  (`priorMapAligned_` guard). Multiple prior maps or repeated re-alignment are not
  supported.
- **Config compatibility is assumed**, not enforced beyond the single-distortion
  / BRISK2 assertions inherited from `Component`. A prior map recorded with a
  different camera system / IMU is undefined.
- **`loadMap` runs while the optimisation threads exist but are idle.** It relies
  on the threads blocking on empty queues; it is not guarded by a mutex.
- **ROS2 node** does not yet expose `--load-map` (only the synchronous app does).

---

## 13. How to use it

**Record a map** (unchanged workflow; output is now `.db`):

```
./okvis_app_synchronous config/euroc.yaml /path/to/V1_01
# → /path/to/okvis2-slam-final_map.db   (written when final BA is enabled)
```

**Relocalize & continue mapping against it:**

```
./okvis_app_synchronous config/euroc.yaml /path/to/V1_02 \
    --load-map /path/to/okvis2-slam-final_map.db
# → loads the prior map (frozen anchor), relocalizes the live run onto it,
#   continues mapping, and writes a combined okvis2-slam-final_map.db
```

**Inspect a map:**

```
sqlite3 okvis2-slam-final_map.db ".tables"
sqlite3 okvis2-slam-final_map.db "SELECT count(*) FROM states;"
sqlite3 okvis2-slam-final_map.db "SELECT count(*) FROM landmarks;"
```

---

## 14. Future work

- Behavioural validation on EuRoC/TUM-VI (the four tests in §11).
- A standalone round-trip unit test (load a saved `.db`, assert structural
  identity) so Layer A is regression-tested without a dataset.
- "Frozen + pose-graph" policy option to bound prior-map memory.
- Persist `ImuParameters` / `NCameraSystem` fully in `meta` for a self-contained,
  config-free reload (currently re-supplied from config and stored only for
  provenance).
- Drop unused prior reprojection residuals from the ceres problem (keep only the
  constant landmark/pose anchors) to lighten optimisation when a large prior map
  is loaded.
- Multi-map / repeated re-alignment support; `--load-map` for the ROS2 node.
```

