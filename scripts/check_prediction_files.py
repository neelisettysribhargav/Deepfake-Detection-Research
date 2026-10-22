"""
check_prediction_files.py

Integrity check for the per-image prediction files, to run BEFORE aggregate_study.py:

  python scripts/check_prediction_files.py
  python scripts/check_prediction_files.py --tags cswin_A_s1 cswin_B_s1 cswin_C_s1

For every run it verifies
  * official predictions  (experiments/predictions/<tag>_predictions.csv):
      6 fields per line; no duplicate (split, path); expected rows
      validation 39428, test 10905, train_sample 10000 (override with --expect-*)
  * robustness predictions (experiments/robustness/<tag>_robust_predictions.csv):
      7 fields per line; no duplicate (condition, path); every condition present;
      every condition has the same number of rows as the pool (DFDC: its own size);
      finite logits.
It tolerates the space padding some editors add after commas. Exit code 1 if any file is bad.
A file can be damaged by two processes writing the same path, an interrupted run, or a
copy/paste upload; if a file fails, re-run the evaluation for that tag.
"""

import os
import re
import csv
import sys
import glob
import math
import argparse
from collections import Counter

TYPES = ["lighting", "color_cast", "gaussian_blur", "block_occlusion", "gaussian_noise", "jpeg",
         "salt_pepper", "pixelate", "motion_blur", "color_distort", "overlay_occluder"]


def read_rows(path, n_fields):
    rows, bad = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader, None)
        for i, r in enumerate(reader, start=2):
            if len(r) != n_fields:
                bad.append((i, len(r)))
                continue
            rows.append([x.strip() for x in r])
    return header, rows, bad


def check_official(path, expect):
    problems = []
    header, rows, bad = read_rows(path, 6)
    if bad:
        problems.append(f"{len(bad)} malformed lines (first at line {bad[0][0]}, {bad[0][1]} fields)")
    cnt = Counter(r[0] for r in rows)
    for split, n in expect.items():
        if cnt.get(split, 0) != n:
            problems.append(f"split '{split}': {cnt.get(split, 0)} rows, expected {n}")
    dup = len(rows) - len({(r[0], r[1]) for r in rows})
    if dup:
        problems.append(f"{dup} duplicate (split, path) rows")
    return problems


def check_robust(path, expect_conditions):
    problems = []
    header, rows, bad = read_rows(path, 7)
    if bad:
        problems.append(f"{len(bad)} malformed lines (first at line {bad[0][0]}, {bad[0][1]} fields)")
    cnt = Counter(r[0] for r in rows)
    pool_n = cnt.get("pool", 0)
    if pool_n == 0:
        problems.append("no 'pool' rows")
    for c in expect_conditions:
        if c not in cnt:
            problems.append(f"missing condition '{c}'")
        elif c != "dfdc_clean" and cnt[c] != pool_n:
            problems.append(f"condition '{c}': {cnt[c]} rows, pool has {pool_n}")
    dup = len(rows) - len({(r[0], r[4]) for r in rows})
    if dup:
        problems.append(f"{dup} duplicate (condition, path) rows")
    nonfinite = sum(1 for r in rows if not math.isfinite(float(r[6]))) if rows else 0
    if nonfinite:
        problems.append(f"{nonfinite} non-finite logits")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None, help="default: every run found")
    ap.add_argument("--pred-dir", default="experiments/predictions")
    ap.add_argument("--rob-dir", default="experiments/robustness")
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--no-dfdc", action="store_true")
    ap.add_argument("--expect-val", type=int, default=39428)
    ap.add_argument("--expect-test", type=int, default=10905)
    ap.add_argument("--expect-train", type=int, default=10000)
    args = ap.parse_args()

    if args.tags is None:
        files = glob.glob(os.path.join(args.rob_dir, "*_robust_predictions.csv"))
        tags = sorted(os.path.basename(f)[:-len("_robust_predictions.csv")] for f in files)
    else:
        tags = args.tags
    conds = ["pool"] + [f"{t}@{s}" for t in TYPES for s in args.severities] + ([] if args.no_dfdc else ["dfdc_clean"])
    expect = {"validation": args.expect_val, "test": args.expect_test, "train_sample": args.expect_train}

    any_bad = False
    for tag in tags:
        for kind, path, fn in (
                ("official", os.path.join(args.pred_dir, f"{tag}_predictions.csv"), lambda p: check_official(p, expect)),
                ("robust", os.path.join(args.rob_dir, f"{tag}_robust_predictions.csv"), lambda p: check_robust(p, conds))):
            if not os.path.isfile(path):
                print(f"[MISSING] {tag} {kind}: {path}")
                any_bad = True
                continue
            problems = fn(path)
            if problems:
                any_bad = True
                print(f"[BAD]     {tag} {kind}: " + "; ".join(problems[:6]))
            else:
                print(f"[OK]      {tag} {kind}")
    if any_bad:
        print("\nAt least one file is bad: re-run the evaluation for that tag (delete its metrics files first so the runner redoes it).")
        sys.exit(1)
    print("\nAll prediction files passed.")


if __name__ == "__main__":
    main()
