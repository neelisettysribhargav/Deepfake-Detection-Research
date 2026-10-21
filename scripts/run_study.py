"""
run_study.py

Automated orchestrator for the Phase 1 9-run robustness study across Arms A, B, and C
with seeds 1, 2, and 3.

For each run:
  1. Checks if outputs already exist (idempotent / crash-safe resume).
  2. Runs `scripts/train_arms.py` (--train-fraction 0.5, 3 epochs).
  3. Runs `scripts/evaluate.py` (standard test/validation splits).
  4. Runs `scripts/evaluate_robustness.py` (35-condition paired robustness suite).
  5. Prints and logs summary metrics.

Usage:
  # Run the full remaining queue sequentially:
  python scripts/run_study.py

  # Or run just one arm / seed:
  python scripts/run_study.py --arm A --seed 1
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_RUNS = [
    ("C", 1),   # (Already completed as pilot; will be verified & skipped)
    ("A", 1),
    ("B", 1),
    ("C", 2),
    ("A", 2),
    ("B", 2),
    ("C", 3),
    ("A", 3),
    ("B", 3),
]


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
    tag = f"cswin_{arm}_s{seed}"
    ckpt_dir = os.path.join(ROOT, "experiments", "robust_runs", tag)
    last_pth = os.path.join(ckpt_dir, "last.pth")
    pred_metrics = os.path.join(ROOT, "experiments", "predictions", f"{tag}_metrics.csv")
    rob_metrics = os.path.join(ROOT, "experiments", "robustness", f"{tag}_robust_metrics.csv")

    print("=" * 80)
    print(f"STARTING STUDY RUN: Arm {arm} | Seed {seed} | Tag: {tag}")
    print("=" * 80)

    # 1. Training
    if os.path.isfile(last_pth):
        print(f"Checkpoint already exists at {last_pth}. Skipping training.")
    else:
        train_cmd = [
            PYTHON, "scripts/train_arms.py",
            "--arm", arm,
            "--seed", str(seed),
            "--weights", weights,
            "--train-fraction", str(train_fraction),
        ]
        run_cmd(train_cmd, f"Training Arm {arm} (seed {seed})")

    # 2. Standard Evaluation (test / validation splits)
    if os.path.isfile(pred_metrics):
        print(f"Standard metrics already exist at {pred_metrics}. Skipping evaluate.py.")
    else:
        eval_cmd = [
            PYTHON, "scripts/evaluate.py",
            "--checkpoint", last_pth,
            "--tag", tag,
            "--model", "CSWin_64_24322_small_256",
            "--dedup-manifest", "data/manifests/author_data_val_dedup.csv",
        ]
        run_cmd(eval_cmd, f"Evaluating standard splits for {tag}")

    # 3. Robustness Evaluation (35 conditions)
    if os.path.isfile(rob_metrics):
        print(f"Robustness metrics already exist at {rob_metrics}. Skipping evaluate_robustness.py.")
    else:
        robust_cmd = [
            PYTHON, "scripts/evaluate_robustness.py",
            "--checkpoint", last_pth,
            "--tag", tag,
            "--model", "CSWin_64_24322_small_256",
            "--pool", "data/manifests/robust_pool.csv",
            "--dfdc", "data/manifests/dfdc_faces.csv",
        ]
        run_cmd(robust_cmd, f"Evaluating 35 robustness conditions for {tag}")

    print(f"\n[DONE] Run {tag} fully finished!")
    print(f"  Standard metrics:   {pred_metrics}")
    print(f"  Robustness metrics: {rob_metrics}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Study Matrix")
    parser.add_argument("--arm", choices=["A", "B", "C"], default=None, help="Run only this arm")
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], default=None, help="Run only this seed")
    parser.add_argument("--weights", default="checkpoints/cswin/cswin_small_224.pth", help="Pretrained weights")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="Train fraction per epoch (0.5 = ~24 min/ep)")
    args = parser.parse_args()

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
    print(f"ALL SCHEDULED STUDY RUNS COMPLETED in {total_elapsed:.1f} minutes ({total_elapsed/60.0:.2f} hours).")
    print("=" * 80)


if __name__ == "__main__":
    main()
