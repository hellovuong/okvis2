# Track 4 (live backend) — Delayed Marginalization + Marginalization Replacement on Ceres

> Scope addendum reflecting the current state after the DM-VIO/GTSAM work.
> Status: **scoped, not started.** Risk: **highest of the project.** Effort ~1800 LoC.

## Why this is the real DM-VIO win for OKVIS

The dynamic-init contribution (Task B) gave ~10% ATE on room1/2 but is inherently
small here: OKVIS already keeps gravity free at init and stereo gives scale. DM-VIO's
*second* contribution — a **relinearizable marginalization prior** — is where OKVIS
genuinely lacks capability and where the long-run / loop-closure / bias-drift gains live.

## The gap (verified in code)

OKVIS's live marginalization is **destructive and non-relinearizable**:
- `ViGraphEstimator::convertToPoseGraphMst` converts visual observations between
  keyframes into a 6-DoF `TwoPoseGraphError` (relative-pose factor), linearized at
  conversion time, then **deletes** the original observations. IMU bias at conversion
  is baked in and never re-evaluated.
- So when biases drift, or loop closure corrects states, the pose-graph edges keep the
  stale linearization — invisible error baked into the trajectory.

What we built (and validated) on `GtsamBackend` is exactly the fix, but it is **not
live**: `DelayedGraph` (retains raw factors for a lag window) + `recomputeBoundaryPrior`
(re-derive the separator prior at corrected linearization) + `remarginalize` /
`maybeRemarginalize` (bias-change trigger, throttle, circuit-breaker). These run only
on the GTSAM backend, which is not wired into `ThreadedSlam`.

## Milestone-2 finding (2026-06-15): observation preservation ALREADY EXISTS

Investigating milestone 2 ("preserve observations") revealed OKVIS already has the
substrate, so milestones 2 & 4 collapse into composing existing primitives:

- `TwoPoseGraphError` **retains its observations** internally (`observations_`, plus
  per-edge `landmarks_` in the reference frame) and is **reversible**:
  `convertToReprojectionErrors()` rebuilds the full reprojection observations with
  landmark positions recovered at the **current** reference pose
  (`hp_W = T_WS0 * landmarks_[id]`) — i.e. re-expressed at the corrected linearization.
- `ViGraphEstimator::convertToObservations(keyframe, …)` already un-marginalizes a
  keyframe's pose-graph edges back into live observations + landmarks, and is used in
  production by loop closure (`ViSlamBackend.cpp:369, 1331`).
- `convertToPoseGraphMst(…)` re-marginalizes.

So **marginalization replacement = `convertToObservations` (un-marg) followed by
`convertToPoseGraphMst` (re-marg) at the current poses** — both production-tested.
No new observation-preservation code is needed; the bias/linearization staleness is
fixed by round-tripping an edge through these two existing methods.

> Note: `TwoPoseGraphError` is **visual-only** (relative pose, landmarks marginalized);
> IMU bias does not enter it. What goes stale is its relative-pose **linearization
> point** after loop-closure / large pose corrections. The IMU edges (`ImuError`)
> already re-preintegrate on bias change. So the re-marg trigger is primarily
> pose-correction / loop-closure driven, not bias-driven.

### Revised milestones (simplified)
- **M2 (done — finding):** preservation confirmed in `TwoPoseGraphError` + `convertToObservations`.
- **M3:** lag `fullGraph_` (or gate which pose-graph KFs are eligible for re-marg).
- **M4:** `remarginalisePoseGraph(keyframe)` = `convertToObservations(kf)` +
  `convertToPoseGraphMst({kf}∪connected)` — thin composition of existing methods.
- **M5:** trigger (post-loop-closure / pose-correction threshold) + throttle; validate
  on real data that the round-trip is non-destructive and improves post-loop-closure ATE.

## Two paths

**Path 1 — Ceres-native (recommended for the live system).** Extend the existing MST
marginalization to be re-derivable. This is the original `track-4-delayed-marginalization.md`
plan, still the pragmatic route since the live backend is Ceres:
1. **Preserve observations in `fullGraph_`** for a lag of N keyframes (skip the
   `removeObservation` deletions when converting in the full graph only).
2. **Lag `fullGraph_` behind `realtimeGraph_`** by N KFs — decouple the lag from
   loop-closure state in `synchroniseRealtimeAndFullGraph`.
3. **`remarginalize(firstId,lastId)`** — rebuild the `TwoPoseGraphError` from the
   preserved observations at the *current* bias; swap the Ceres residual block.
4. **Bias-change trigger** — accumulate ‖Δb_g‖+‖Δb_a‖; re-marg over the lag window
   when it exceeds `remarg_bias_threshold`; throttle by `remarg_min_interval_sec`;
   3-strike circuit breaker. Also trigger after `addLoopClosureFrame`.

**Path 2 — GTSAM delayed marg live.** Reuse the tested `DelayedGraph`/`remarginalize`
directly — but that requires `GtsamBackend` to *be* the live backend (the full
`ViSlamBackend`→`GtsamBackend` swap: ~75 frontend/backend methods incl. loop closure,
landmark lifecycle, covisibility). Out of scope until that swap is done; the GTSAM
components stand as the reference implementation + validation for when it is.

## First milestones (Path 1)

1. **Gating prototype:** confirm Ceres supports mid-optimization
   `RemoveResidualBlock` + `AddResidualBlock` without corrupting solver state
   (unit test). This gates the whole design — if it doesn't hold, re-marg must rebuild
   the affected sub-problem (cost +~30%).
2. `convertToPoseGraphMst` gains a `preserveObservations` flag (fullGraph_ only).
3. Lag logic in `synchroniseRealtimeAndFullGraph` (`delayed_marginalization_lag`, the
   `ImuParameters` field already added/parsed).
4. `ViGraphEstimator::remarginalize` rebuilding `TwoPoseGraphError` at current bias.
5. Bias-change tracker + trigger (params `remarg_bias_threshold`,
   `remarg_min_interval_sec` already added/parsed) + circuit breaker.

## Risks

- **Ceres residual swap mid-solve** (milestone 1) — biggest unknown.
- **Observation memory lifecycle** — preserve only within the lag; aggressive cleanup
  past it (audit `cleanUnobservedLandmarks`).
- **`TwoPoseGraphError` immutability** — rebuild, never edit in place.
- **Diminishing returns on short sequences** — room1/2 won't show it (well-excited,
  short). Needs a **long-run / low-excitation benchmark** (concatenated TUM-VI or a
  highway-style recording) to demonstrate the gain. Pass criterion: long-run ATE down
  ≥10% vs baseline; no regression (<2%) on short EuRoC/room sequences; frame-time p99
  within budget.

## Reusable infrastructure already present

- `touchedStates_`/`touchedLandmarks_` cross-graph batching (`ViSlamBackend`).
- `ImuError::redoPreintegration` for bias-corrected re-evaluation.
- `isLoopClosing_` race guard.
- The validated GTSAM delayed-marg components as the reference for correctness checks.
