# Track 5 (project) — Full `ViSlamBackend` → `GtsamBackend` swap

> Standalone project. Size: **large (multi-month)**. Risk: **high**. Prereq: the DM-VIO
> GTSAM components (built + unit-tested on `feature/dmvio-gtsam-backend`).
> Decision gate up front — see "Go / no-go".

## Goal

Make the GTSAM factor-graph backend the **live** optimisation backend driving
`ThreadedSlam`, so the validated DM-VIO components run end-to-end on real data:
- **true visual-prior PGBA** initialization (not just the inertial-only live path),
- **GTSAM delayed marginalization + marginalization replacement** (relinearizable prior),
- optional **ISAM2** incremental optimisation,
all behind a config/compile flag, with the Ceres `ViSlamBackend` retained as default
until parity is proven.

## Why this is a project, not a feature

`ViSlamBackend`/`ViGraph(Estimator)` is not a thin solver wrapper — it *is* the SLAM
back-end. The coupling that must be reproduced:
- **~70 public `ViSlamBackend` methods**; **45 distinct estimator calls from `Frontend`**
  (incl. the **templated** `addObservation<CAMERA_GEOMETRY>` — blocks a clean virtual
  interface, forces compile-time selection).
- **Dual graph** (`realtimeGraph_` + async `fullGraph_`) with cross-graph batching
  (`touchedStates_`/`touchedLandmarks_`).
- **Loop closure + place recognition** (DBoW2, `attemptLoopClosure`, `addLoopClosureFrame`,
  un-marg `convertToObservations` → full-graph re-opt → re-marg), pose **freezing**
  (`freezePosesUntil`, `minDeltaT`), keyframe/IMU-frame **window strategy** (`applyStrategy`).
- **Landmark lifecycle** (`addLandmark`/`setLandmark`/`mergeLandmark(s)`/
  `cleanUnobservedLandmarks`/covisibility) tightly bound to the frontend's data association.
- **Map serialization** (`saveMap`, `Component` g2o load/save) + **multi-session**
  (track-2) + **online extrinsics**.

GtsamBackend today implements ~12 methods (states, IMU/reproj factors, batch LM,
Schur marginalisation, delayed-graph, applyDynamicInitialisation). The swap is bringing
it to functional parity with the back-end's *role*, not re-deriving math (that's done).

## Already built (reuse — DM-VIO work, unit-tested)

| Asset | Role in the swap |
|---|---|
| `ImuPreintegrationGtsam`, `GtsamConversions` | IMU edges, state mapping (bias-flip pinned) |
| `GtsamReprojectionFactor<GEOMETRY>` | visual edges (reuses OKVIS camera models) |
| `Marginalization::marginalizeOut` + `LinearContainerFactor` | sliding-window marg prior |
| `DelayedGraph` (advance/recomputeBoundaryPrior) | delayed marg + replacement |
| `GravityAlignmentFactor`, `PgbaImuFactor`, `ViImuInitializer` | dynamic init + true PGBA |
| `GtsamBackend` (batch LM, marginalizeKeys, remarginalize, rewriteAfterInit) | core graph backend |

## Capability gap (GtsamBackend vs the ViSlamBackend role)

| Capability | GtsamBackend | Needed |
|---|---|---|
| Add states / IMU / reprojection / marginalise / solve | ✅ | — |
| Templated `addObservation<GEOMETRY>` matching frontend call sites | partial | full signature parity |
| Keyframe/IMU-frame **window strategy** (`applyStrategy`) | ❌ | port the policy |
| **Landmark lifecycle** (set/merge/clean/covisibility) | basic add | full parity |
| **Loop closure + place recognition** (DBoW, un-marg/re-marg) | ❌ | the hard part |
| Pose **freezing** / fixed-lag semantics | ❌ | port |
| **Dual graph** realtime/full or single-graph + ISAM2 | single batch | decide + implement |
| **Map save/load** (`Component`) + multi-session | ❌ | port serialization |
| Online extrinsics | prior-anchored | parity |
| Trajectory output / state callbacks | getters | wire to `ThreadedSlam` |

## Key architecture decisions (resolve in S0)

1. **Selection mechanism:** compile-time `using BackendType = …` in `ThreadedSlam.hpp`
   (the templated `addObservation` rules out a runtime virtual interface). Two builds to
   A/B. → recommended.
2. **Graph topology:** (a) mirror OKVIS's dual realtime/full graph, or (b) single graph +
   `ISAM2` incremental + fixed-lag smoother. ISAM2 is the more "GTSAM-native" path and the
   closer analogue to DM-VIO's incremental scheme, but diverges from OKVIS's loop-closure
   model. **Recommend starting with batch + explicit marg (already built) to reach VIO
   parity, then evaluate ISAM2 as an optimisation.**
3. **Landmark representation:** `Point3` (Euclidean, fine for stereo) vs a 4D homogeneous
   manifold (OKVIS-faithful, handles points at infinity). Start `Point3`; revisit if
   far-point count matters.
4. **Loop-closure model:** reuse OKVIS's reversible un-marg/re-marg cycle (it maps onto
   `DelayedGraph`) vs a GTSAM pose-graph + switchable constraints. Reuse OKVIS's model.

## Phased plan (each phase gated by parity vs the Ceres baseline)

- **S0 — Architecture + parity harness (~1–2 wk).** Decide 1–4 above. Promote the ad-hoc
  eval (`/tmp/eval_rooms.sh`, `to_tum.py`) into `evaluation/` (evo ATE vs mocap, room1–6 +
  EuRoC). Define the pass band (ATE within run-to-run noise of Ceres).
- **S1 — VIO core parity, no loop closure (~3–4 wk).** GtsamBackend implements the full
  windowed VIO loop driving `ThreadedSlam` (states, frontend `addObservation`, window
  strategy, marginalisation, freezing, getters/trajectory). Gate: ATE parity on room1–6 /
  EuRoC **VO+IMU without loop closure**.
- **S2 — Landmark lifecycle + data association (~2–3 wk).** `setLandmark`/`merge`/
  `cleanUnobserved`/covisibility parity with the frontend. Gate: landmark counts + ATE
  match Ceres within noise.
- **S3 — Loop closure + true PGBA init + delayed marg live (~4–6 wk).** DBoW place
  recognition, `attemptLoopClosure`/`addLoopClosureFrame` via `DelayedGraph` un-marg/re-marg;
  wire `ViImuInitializer` with the **visual marginalization prior** (true PGBA) +
  `rewriteAfterInit`. Gate: loop-closure success rate + post-loop ATE ≥ Ceres.
- **S4 — Map save/load + multi-session + online extrinsics (~2–3 wk).** Port `Component`
  serialization; relocalisation parity. Gate: save→reload→relocalise works.
- **S5 — Integration, defaults, full benchmark gate (~1–2 wk).** Default backend stays
  Ceres; `backend: gtsam` opt-in. Full EuRoC + TUM-VI parity table; ARM frame-time p99.

## Risks

1. **Loop closure (S3) is the crux** — un-marg/re-marg + DBoW + dual-graph re-sync is the
   most complex and least testable-in-isolation. Most schedule risk lives here.
2. **Performance on ARM** — GTSAM batch/ISAM2 + the templated visual factors vs Ceres'
   tuned sparse solver; must hold the realtime budget (`realtime_time_limit`).
3. **Templated frontend call sites** → compile-time selection → two builds; CI cost.
4. **Map-format compatibility** — existing `.db`/g2o maps + the room `*_prior.db`; either
   port the format or accept a new one (breaks multi-session interop).
5. **Eigen/Boost/TBB footprint** of GTSAM on the embedded target (already mitigated: shared
   system Eigen).
6. **Parity is fuzzy** — OKVIS run-to-run nondeterminism (~±1 mm ATE on rooms) means gates
   must be noise-band, not exact; needs multi-trial averaging.

## Effort & honest expectation

Order **2–4 engineer-months**. The *algorithmic* novelty is already done and tested; this
project is **integration + parity engineering** against a large, coupled API. Expected
accuracy outcome: **parity with Ceres** on the well-conditioned sequences (not a win there
— OKVIS is already strong), with the real upside being (a) live **true PGBA** init on
hard/fast-start sequences and (b) a GTSAM-native foundation for future work (ISAM2,
photometric/direct factors, tighter DM-VIO fidelity).

## Go / no-go recommendation

Do this **only if** the goal is a GTSAM-native backend for its own sake (future direct/
photometric factors, ISAM2, research fidelity to DM-VIO) — **not** for near-term ATE on
EuRoC/TUM-VI, where the Ceres backend already matches and the DM-VIO init contribution is
already live (~10% where init is poor). If the objective is "best OKVIS accuracy now," the
current Ceres backend + the landed dynamic-init feedback is the better investment, and this
project can be deferred. Start at **S0** (architecture decision + parity harness) before
committing to S1+.
