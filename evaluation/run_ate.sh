#!/usr/bin/env bash
# Parity harness (track-5 S0): run okvis_app_synchronous over TUM-VI sequences with
# a given config, N trials each, and compute ATE RMSE vs mocap ground truth (evo,
# SE3-aligned). Multiple trials characterize OKVIS's run-to-run nondeterminism so
# parity can be judged against a noise band rather than an exact number.
#
# Usage:
#   evaluation/run_ate.sh --config <yaml> --label <name> [options]
# Options (env-overridable):
#   --seqs   "room1 room2 ..."   sequences (default: room1..room6)
#   --trials N                   trials per sequence (default: 3)
#   --out    DIR                 results dir (default: evaluation/results)
#   --data-root DIR              dataset root (default: $DATA_ROOT or /home/virtual/rosbags)
#   --gt-rel PATH                GT csv relative to mav0 (default: mocap0/data.csv)
# Env: OKVIS_APP (app path), ROS_SETUP (default /opt/ros/jazzy/setup.bash),
#      GTSAM_LIB (default /usr/local/lib), DATASET_SUFFIX (default _512_16)
#
# Output: <out>/<label>.csv with columns: label,seq,trial,rmse_m  + a per-seq summary.
# No 'set -e/-u/pipefail': we source ROS setup (which references unset vars) and run
# many pipes/commands that may fail per-trial; failures are handled explicitly.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$HERE/../../.." && pwd)"   # .../workspace

SEQS="room1 room2 room3 room4 room5 room6"
TRIALS=3
OUT="$HERE/results"
DATA_ROOT="${DATA_ROOT:-/home/virtual/rosbags}"
GT_REL="mocap0/data.csv"
SUFFIX="${DATASET_SUFFIX:-_512_16}"
CONFIG=""; LABEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --seqs) SEQS="$2"; shift 2;;
    --trials) TRIALS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --gt-rel) GT_REL="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$CONFIG" ] && [ -n "$LABEL" ] || { echo "need --config and --label"; exit 2; }

source "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}" 2>/dev/null || true
export LD_LIBRARY_PATH="${GTSAM_LIB:-/usr/local/lib}:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"
APP="${OKVIS_APP:-}"
if [ -z "$APP" ]; then
  APP="$(find "$WS_ROOT/build" "$WS_ROOT/install" -name okvis_app_synchronous -type f -print -quit 2>/dev/null)"
fi
[ -n "$APP" ] && [ -x "$APP" ] || { echo "okvis_app_synchronous not found (set OKVIS_APP)"; exit 1; }

mkdir -p "$OUT"
CSV="$OUT/$LABEL.csv"
echo "label,seq,trial,rmse_m" > "$CSV"
echo "harness: app=$APP config=$CONFIG label=$LABEL trials=$TRIALS"

for seq in $SEQS; do
  DS="$DATA_ROOT/dataset-${seq}${SUFFIX}/mav0"
  GT="$DS/$GT_REL"
  [ -d "$DS" ] || { echo "  skip $seq (no $DS)"; continue; }
  python3 "$HERE/traj_to_tum.py" "$GT" "$OUT/${seq}_gt.txt" mocap >/dev/null
  for t in $(seq 1 "$TRIALS"); do
    GLOG_logtostderr=1 timeout 900 "$APP" "$CONFIG" "$DS" >/dev/null 2>&1 || true
    RMSE="nan"
    if [ -f "$DS/okvis2-slam-final_trajectory.csv" ]; then
      cp "$DS/okvis2-slam-final_trajectory.csv" "$OUT/${LABEL}_${seq}_t${t}.csv"
      python3 "$HERE/traj_to_tum.py" "$OUT/${LABEL}_${seq}_t${t}.csv" "$OUT/${LABEL}_${seq}_t${t}.txt" okvis >/dev/null 2>&1
      R=$(evo_ape tum "$OUT/${seq}_gt.txt" "$OUT/${LABEL}_${seq}_t${t}.txt" -a --t_max_diff 0.02 2>/dev/null | awk '/rmse/{print $2}')
      [ -n "$R" ] && RMSE="$R"
    fi
    echo "$LABEL,$seq,$t,$RMSE" >> "$CSV"
    echo "  $seq trial $t: rmse=$RMSE m"
  done
done
echo "=== summary ($LABEL) ==="
python3 "$HERE/parity.py" --summary "$CSV"
echo "wrote $CSV"
