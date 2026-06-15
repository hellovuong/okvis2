#!/usr/bin/env python3
"""Summarize / compare ATE parity results produced by run_ate.sh.

A results CSV has rows: label,seq,trial,rmse_m

Summary:   parity.py --summary results/<label>.csv
Compare:   parity.py --baseline results/ceres.csv --candidate results/gtsam.csv

Compare judges parity against a noise band: OKVIS is nondeterministic, so a per-seq
difference is only meaningful if it exceeds the combined run-to-run spread. A
candidate PASSES a sequence if its mean RMSE is within max(0.5mm, baseline_std +
candidate_std) of the baseline mean (i.e. inside the noise band), or better.
"""
import argparse
import csv
import math
from collections import defaultdict


def load(path):
    by_seq = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                by_seq[row['seq']].append(float(row['rmse_m']))
            except (ValueError, KeyError):
                pass
    return by_seq


def stats(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float('nan'), float('nan'), 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def fmt_mm(x):
    return "nan" if math.isnan(x) else f"{x*1000:.2f}"


def summary(path):
    by_seq = load(path)
    print(f"{'seq':<8} {'mean[mm]':>9} {'std[mm]':>8} {'n':>3}")
    for seq in sorted(by_seq):
        m, sd, n = stats(by_seq[seq])
        print(f"{seq:<8} {fmt_mm(m):>9} {fmt_mm(sd):>8} {n:>3}")


def compare(base_path, cand_path):
    b, c = load(base_path), load(cand_path)
    seqs = sorted(set(b) | set(c))
    print(f"{'seq':<8} {'base[mm]':>10} {'cand[mm]':>10} {'delta[mm]':>10} {'band[mm]':>9}  verdict")
    n_pass = 0
    n_total = 0
    for seq in seqs:
        bm, bsd, bn = stats(b.get(seq, []))
        cm, csd, cn = stats(c.get(seq, []))
        if math.isnan(bm) or math.isnan(cm):
            print(f"{seq:<8} {fmt_mm(bm):>10} {fmt_mm(cm):>10} {'-':>10} {'-':>9}  MISSING")
            continue
        n_total += 1
        delta = cm - bm
        band = max(0.0005, bsd + csd)  # 0.5mm floor + combined spread
        verdict = "PASS" if delta <= band else "FAIL"
        if verdict == "PASS":
            n_pass += 1
        tag = " (better)" if delta < -band else ""
        print(f"{seq:<8} {fmt_mm(bm):>10} {fmt_mm(cm):>10} {fmt_mm(delta):>10} "
              f"{fmt_mm(band):>9}  {verdict}{tag}")
    print(f"\nparity: {n_pass}/{n_total} sequences within noise band (or better)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary')
    ap.add_argument('--baseline')
    ap.add_argument('--candidate')
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    elif a.baseline and a.candidate:
        compare(a.baseline, a.candidate)
    else:
        ap.error("use --summary CSV, or --baseline CSV --candidate CSV")
