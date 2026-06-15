# ADR-0001 — GTSAM backend architecture (closes track-5 S0)

Status: **accepted** · Date: 2026-06-15 · Context: track-5 (full backend swap).

## Decisions

1. **Backend selection = compile-time, via the existing `okvis::Estimator` typedef.**
   `Frontend` already operates on `okvis::Estimator&` (typedef in
   `okvis_common/include/okvis/ViFrontendInterface.hpp:65`, currently
   `= ViSlamBackend`). A new build option `OKVIS_USE_GTSAM_BACKEND` (default **OFF**)
   switches it to `GtsamBackend`. Runtime/virtual selection is **rejected**: the
   frontend calls the **templated** `addObservation<CAMERA_GEOMETRY>`, which cannot
   go through a virtual interface.
   - **Dependency wrinkle (must resolve in S1):** the typedef lives in `okvis_common`,
     which must **not** depend on `okvis_ceres` (where `GtsamBackend` lives) — that
     would be circular. Resolution: relocate the selection typedef into an
     `okvis_ceres` header (e.g. `okvis/Estimator.hpp`) included by `Frontend`/
     `ThreadedSlam`, and have `ViFrontendInterface` forward-declare only. `ThreadedSlam`
     must also change its member from the concrete `ViSlamBackend estimator_` to
     `okvis::Estimator estimator_`, and the `optimiseFullGraph` thread-bind must not
     hard-reference `ViSlamBackend::` (use `Estimator::`).

2. **Optimisation topology = batch + explicit marginalisation first; ISAM2 later.**
   Reuse the built, tested components (`GtsamBackend` batch LM + `Marginalization` +
   `DelayedGraph`). Mirror OKVIS's role incrementally: a single active graph for S1
   (VIO, no loop closure); add the full/delayed graph in S3 (loop closure). ISAM2 is
   an optimisation evaluated only after batch parity is reached.

3. **Landmarks = `gtsam::Point3`** (Euclidean; correct for the stereo target). Revisit
   a 4D homogeneous manifold only if far/at-infinity points prove material.

4. **Loop closure = reuse OKVIS's reversible model** (`convertToObservations` un-marg →
   full-graph re-opt → `convertToPoseGraphMst` re-marg), which maps onto `DelayedGraph`.
   No switchable-constraint pose-graph rewrite.

5. **Parity gate = the S0 multi-trial noise-band harness** (`evaluation/`). Each phase
   passes only if the candidate's ATE is within `max(0.5mm, base_std+cand_std)` of the
   Ceres baseline (or better), multi-trial. (S0 already showed single-run deltas on
   rooms are noise.)

6. **Ceres `ViSlamBackend` stays the default/production backend** until full parity
   (S5). `OKVIS_USE_GTSAM_BACKEND` is opt-in and incomplete until then.

## S1 work breakdown — the `Estimator` API surface

`GtsamBackend` must implement the methods `Frontend`+`ThreadedSlam` call on
`okvis::Estimator` (~60 real methods; a few grep hits like `do_loop_closures`,
`p_dbow`, `realtime_max_iterations` are `parameters_.estimator.FIELD`, not backend
methods). Have (5): `addLandmark`, `addObservation`, `extrinsics`, `getLandmark`,
`pose`. Grouped TODO:

- **State/window (S1 core):** `addStates`, `addCamera`, `addImu`, `numFrames`,
  `currentStateId`, `stateIdByAge`, `timestamp`, `isKeyframe`, `setKeyframe`,
  `isInImuWindow`, `imuFrames`, `keyFrames`, `applyStrategy`, `optimiseRealtimeGraph`,
  `setOptimisation/OptimizationTimeLimit`, `setPose`, `speedAndBias`, `setDetectorUniformityRadius`.
- **Landmark/observation (S2):** `getLandmarks`, `getObservedIds`, `isObserved`,
  `isLandmarkAdded`, `isLandmarkInitialised`, `setLandmark`, `setLandmarkInitialized`,
  `setLandmarkClassification`, `setObservationInformation`, `removeObservation`,
  `mergeLandmark(s)`, `cleanUnobservedLandmarks`, `areLandmarksInFrontOfCameras`,
  `multiFrame`, `trackingQuality`, `overlapFraction`, `mostOverlappedStateId`.
- **Loop closure / full graph (S3):** `attemptLoopClosure`, `addLoopClosureFrame`,
  `isLoopClosing`, `isLoopClosureAvailable`, `isLoopClosureFrame`,
  `isRecentLoopClosureFrame`, `isPlaceRecognitionFrame`, `isPoseGraphFrame`,
  `loopClosureFrames`, `needsFullGraphOptimisation`, `synchroniseRealtimeAndFullGraph`,
  `closedLoop`, `T_AiS_`, `applyInitialisation` (already built, wire it).
- **Map / output (S4):** `saveMap`, `writeFinalCsvTrajectory`, `doFinalBa`,
  `drawOverheadImage`, `clear`.

S1 = implement the **State/window** group + drive `ThreadedSlam` for VIO with loop
closure disabled, gated by the S0 harness on room1–6 / EuRoC.
