"""
run_study.py  (corrected version)

Automated orchestrator for the Phase 1 9-run robustness study across Arms A, B, and C
with seeds 1, 2, and 3.

For each run:
  1. Trains (scripts/train_arms.py, --train-fraction 0.5, 3 epochs) unless a COMPLETE run exists.
  2. Evaluates standard splits (scripts/evaluate.py).
  3. Evaluates the 35-condition paired robustness suite (scripts/evaluate_robustness.py).

Fixes compared with the first version:
  * "Training finished" now means history.csv has all epochs. train_arms.py saves last.pth after
    EVERY epoch, so checking only for last.pth would silently evaluate a half-trained model after
    a crash, sleep or power cut. An incomplete run is retrained from scratch (train_arms has no resume).
  * The corruption library must be byte-identical to the frozen copy
    (robustness/corruptions_frozen_phase1.py), checked before every job.

Usage:
  python scripts/run_study.py                    # full remaining queue
  python scripts/run_study.py --arm A --seed 1   # one job
"""

import os
import sys
import time
import hashlib
import argparse
import subprocess
from datetime import datetime

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EPOCHS = 3

DEFAULT_RUNS = [
    ("C", 1),   # registered from the pilot; verified and skipped
    ("A", 1),
    ("B", 1),
    ("C", 2),
    ("A", 2),
    ("B", 2),
    ("C", 3),
    ("A", 3),
    ("B", 3),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_library_frozen():
    cur = os.path.join(ROOT, "robustness", "corruptions.py")
    frozen = os.path.join(ROOT, "robustness", "corruptions_frozen_phase1.py")
    if sha256(cur) != sha256(frozen):
        sys.exit("ERROR: robustness/corruptions.py differs from corruptions_frozen_phase1.py. "
                 "The corruption library is frozen; restore it (git checkout phase1_library_frozen -- robustness/corruptions.py).")


def training_complete(ckpt_dir, epochs=EPOCHS):
    last = os.path.join(ckpt_dir, "last.pth")
    hist = os.path.join(ckpt_dir, "history.csv")
    if not (os.path.isfile(last) and os.path.isfile(hist)):
        return False
    with open(hist, "r", encoding="utf-8") as f:
        rows = [ln for ln in f if ln.strip()]
    return (len(rows) - 1) >= epochs          # minus header


def run_cmd(cmd, desc):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> {desc}")
    print(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    elapsed = (time.time() - t0) / 60.0
    if ret.returncode != 0:
        print(f"ERROR: {desc} failed with exit code {ret.returncode} after {elapsed:.1f} min.")
        sys.exit(ret.returncode)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed in {elapsed:.1f} min.")


def process_run(arm, seed, weights, train_fraction):
    check_library_frozen()
    tag = f"cswin_{arm}_s{seed}"
    ckpt_dir = os.path.join(ROOT, "experiments", "robust_runs", tag)
    last_pth = os.path.join(ckpt_dir, "last.pth")
    pred_metrics = os.path.join(ROOT, "experiments", "predictions", f"{tag}_metrics.csv")
    rob_metrics = os.path.join(ROOT, "experiments", "robustness", f"{tag}_robust_metrics.csv")

    print("=" * 80)
    print(f"STARTING STUDY RUN: Arm {arm} | Seed {seed} | Tag: {tag}")
    print("=" * 80)

    if training_complete(ckpt_dir):
        print(f"Complete training found ({EPOCHS} epochs in history.csv). Skipping training.")
    else:
        if os.path.isfile(last_pth):
            print(f"WARNING: {last_pth} exists but history.csv has fewer than {EPOCHS} epochs "
                  f"(interrupted run). Retraining from scratch.")
        train_cmd = [PYTHON, "scripts/train_arms.py", "--arm", arm, "--seed", str(seed),
                     "--weights", weights, "--train-fraction", str(train_fraction)]
        run_cmd(train_cmd, f"Training Arm {arm} (seed {seed})")

    if os.path.isfile(pred_metrics):
        print(f"Standard metrics already exist at {pred_metrics}. Skipping evaluate.py.")
    else:
        eval_cmd = [PYTHON, "scripts/evaluate.py", "--checkpoint", last_pth, "--tag", tag,
                    "--model", "CSWin_64_24322_small_256",
                    "--dedup-manifest", "data/manifests/author_data_val_dedup.csv"]
        run_cmd(eval_cmd, f"Evaluating standard splits for {tag}")

    if os.path.isfile(rob_metrics):
        print(f"Robustness metrics already exist at {rob_metrics}. Skipping evaluate_robustness.py.")
    else:
        robust_cmd = [PYTHON, "scripts/evaluate_robustness.py", "--checkpoint", last_pth, "--tag", tag,
                      "--model", "CSWin_64_24322_small_256",
                      "--pool", "data/manifests/robust_pool.csv", "--dfdc", "data/manifests/dfdc_faces.csv"]
        run_cmd(robust_cmd, f"Evaluating 35 robustness conditions for {tag}")

    print(f"\n[DONE] Run {tag} fully finished!")
    print(f"  Standard metrics:   {pred_metrics}")
    print(f"  Robustness metrics: {rob_metrics}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Study Matrix")
    parser.add_argument("--arm", choices=["A", "B", "C"], default=None)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--weights", default="checkpoints/cswin/cswin_small_224.pth")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    args = parser.parse_args()

    check_library_frozen()
    if args.arm and args.seed:
        queue = [(args.arm, args.seed)]
    elif args.arm:
        queue = [(args.arm, s) for s in [1, 2, 3]]
    elif args.seed:
        queue = [(a, args.seed) for a in ["C", "A", "B"]]
    else:
        queue = DEFAULT_RUNS

    total_t0 = time.time()
    print(f"Study Queue ({len(queue)} runs): {queue}")

    for idx, (arm, seed) in enumerate(queue, 1):
        print(f"\n>>> Running job {idx}/{len(queue)}: Arm {arm}, Seed {seed}")
        process_run(arm, seed, args.weights, args.train_fraction)

    total_elapsed = (time.time() - total_t0) / 60.0
    print("=" * 80)
    print(f"ALL SCHEDULED STUDY RUNS COMPLETED in {total_elapsed:.1f} minutes ({total_elapsed / 60.0:.2f} hours).")
    print("=" * 80)


if __name__ == "__main__":
    main()
