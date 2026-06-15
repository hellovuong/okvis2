# Evaluation / parity harness

ATE parity tooling for the GTSAM-backend swap (track-5 S0) and general regression
checks. Computes Absolute Trajectory Error (RMSE) vs TUM-VI mocap ground truth with
`evo` (SE3-aligned), over multiple trials so OKVIS's run-to-run nondeterminism can be
treated as a noise band rather than a single number.

## Requirements
- `evo` (`pip install evo`) on PATH (`evo_ape`).
- okvis built (`okvis_app_synchronous`); GTSAM runtime libs on `LD_LIBRARY_PATH`
  (`/usr/local/lib`).
- TUM-VI datasets in EuRoC/`mav0` layout with `mocap0/data.csv`.

## Usage
Run a config over sequences (N trials each):
```
evaluation/run_ate.sh --config config/tumvi_slam_512.yaml      --label ceres --trials 3
evaluation/run_ate.sh --config config/tumvi_slam_512_dmvio.yaml --label gtsam --trials 3
```
Defaults: seqs `room1..room6`, data root `/home/virtual/rosbags`, GT `mocap0/data.csv`.
Override with `--seqs "room1 room4"`, `--data-root DIR`, `--gt-rel ...`, `--out DIR`.

Summarize / compare:
```
evaluation/parity.py --summary evaluation/results/gtsam.csv
evaluation/parity.py --baseline evaluation/results/ceres.csv --candidate evaluation/results/gtsam.csv
```
`compare` prints per-sequence mean/delta and a **PASS/FAIL** vs a noise band
(`max(0.5mm, base_std + cand_std)`): a candidate passes if it's within the band or better.

## Files
- `traj_to_tum.py` — OKVIS / mocap CSV → TUM format for evo.
- `run_ate.sh` — run a config over sequences × trials, write `results/<label>.csv`.
- `parity.py` — per-seq summary, or baseline-vs-candidate parity table.

## Notes
- OKVIS writes `okvis2-slam-final_trajectory.csv` into the dataset `mav0` dir; the
  harness copies it per trial into `results/`.
- Use this to gate each track-5 phase: candidate backend must be within the noise band
  of the Ceres baseline (multi-trial), not exact.
