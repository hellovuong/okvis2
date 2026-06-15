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
- **M3 (folded into M5):** originally "lag `fullGraph_` to preserve raw obs" — no longer
  needed: `TwoPoseGraphError` already retains observations for *every* converted edge,
  recoverable any time. M3 is therefore not a preservation/lag data structure; it
  reduces to a **selection + cost policy** — *which* edges to re-derive and *how many*.
  Since the staleness is pose-driven, the selector is the set of keyframes whose poses
  moved (the loop-closure-updated state set, or a per-edge pose-delta threshold),
  **bounded** by a max edges-per-trigger cap and a throttle. That is exactly the M5
  trigger concern, so M3 merges into M5. (Memory note: observations live inside each
  `TwoPoseGraphError` — a cost OKVIS already pays for loop-closure un-marg; re-marg adds
  none.)
- **M4:** `remarginalisePoseGraph(keyframe)` = `convertToObservations(kf)` +
  `convertToPoseGraphMst({kf}∪connected)` — thin composition of existing methods.
- **M5 (incl. ex-M3):** select affected keyframes (loop-closure set / pose-delta),
  cap + throttle, fire after loop closure; validate on real data that the round-trip is
  non-destructive and improves post-loop-closure ATE.

## Conclusion (2026-06-15): track-4 is largely redundant for OKVIS — recommend NOT implementing M4/M5

Carrying the M4 composition analysis to its end shows OKVIS **already has** what DM-VIO's
delayed marginalization / marginalization replacement provides, for the only case where
it matters:

1. **No staleness without a correction.** Old poses are **frozen** after `minDeltaT=2.0 s`
   (`ViSlamBackend.cpp` `freezePosesUntil`/`freezeSpeedAndBiasesUntil`, ~lines 604/761).
   A marginalized keyframe's pose is constant, so its `TwoPoseGraphError` linearization
   point never drifts — there is nothing to re-derive between corrections.
2. **Replacement already happens at the only correction event.** At loop closure,
   `addLoopClosureFrame` calls `convertToObservations` (un-marg) on the affected frames,
   the full graph re-optimizes them, and `applyStrategy` re-runs `convertToPoseGraphMst`
   (re-marg) at the corrected poses. That **is** marginalization replacement, already in
   production.
3. The visual `TwoPoseGraphError` is **bias-independent** (M2 note), and IMU edges
   already `redoPreintegration` on bias change — so there is no separate bias-drift gap.

DM-VIO's marginalization replacement is a big win for systems whose marginalization is
**irreversible** (e.g. DSO's photometric prior). OKVIS's marginalization is **reversible**
(`TwoPoseGraphError` retains observations) and is already re-derived on loop closure.
The incremental value of M4/M5 is therefore small, while the implementation touches the
most delicate machinery in the system (loop-closure un-marg, pose freezing, dual-graph
`auxiliaryStates_` bookkeeping) with no standalone test path — high risk, low reward.

**Recommendation:** close track-4 here. The DM-VIO IMU contributions worth pursuing in
OKVIS are (a) the dynamic init feedback — done, ~10% ATE where init is poor (room4); and
(b) the full `ViSlamBackend`→`GtsamBackend` swap to bring true visual-prior PGBA + GTSAM
delayed-marg live, if/when a GTSAM backend is desired. Everything below is retained for
reference should OKVIS's freezing/loop-closure model change.

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
