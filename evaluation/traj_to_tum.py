#!/usr/bin/env python3
"""Convert OKVIS / TUM-VI CSV trajectories to TUM format for evo.

TUM format: `timestamp_s tx ty tz qx qy qz qw` (space separated).

Modes:
  okvis  - OKVIS trajectory CSV: ts[ns], p(3), q_xyzw(4), [v(3), b_g(3), b_a(3)]
  mocap  - TUM-VI mocap0 / EuRoC ground truth: ts[ns], p(3), q_wxyz(4), [...]

Both inputs may have a header line and be comma- or space-separated.
"""
import sys


def convert(inp, out, mode):
    rows = []
    with open(inp) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#') or ln[0].isalpha():
                continue
            c = [x for x in ln.replace(',', ' ').split() if x]
            if len(c) < 8:
                continue
            ts = float(c[0]) * 1e-9  # ns -> s
            px, py, pz = c[1], c[2], c[3]
            if mode == 'okvis':           # q_xyzw
                qx, qy, qz, qw = c[4], c[5], c[6], c[7]
            else:                          # mocap / euroc: q_wxyz
                qw, qx, qy, qz = c[4], c[5], c[6], c[7]
            rows.append(f"{ts:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}")
    with open(out, 'w') as f:
        f.write("\n".join(rows) + "\n")
    return len(rows)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('okvis', 'mocap'):
        sys.exit("usage: traj_to_tum.py <in.csv> <out.txt> <okvis|mocap>")
    n = convert(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"{sys.argv[3]}: wrote {n} poses -> {sys.argv[2]}")
