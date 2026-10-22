"""
run_ablation.py

Runs the coverage ablation: 2 new arms x 3 seeds = 6 runs (about 100 min each, 10 h total).

  CnoNJ  arm C without Gaussian noise and JPEG in the training sampler
  BN     arm B (generic augmentation) plus Gaussian noise and JPEG at the same frequency as in arm C

Order is interleaved (CnoNJ s1, BN s1, CnoNJ s2, ...) so slow drift affects both arms equally.
Same safety rules as run_study.py: a run counts as finished only when history.csv has all
epochs (an interrupted run is retrained from scratch), the frozen corruption library is checked
before every job, and finished steps are skipped on re-run.

  python scripts/run_ablation.py --dry-run          # print the commands only
  python scripts/run_ablation.py                    # run everything
  python scripts/run_ablation.py --arm BN --seed 2  # one job

Put this file in scripts/ next to run_study.py (it imports its helper functions).
"""

import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_study as rs                                           # noqa: E402

PYTHON, ROOT = rs.PYTHON, rs.ROOT

ARMS = {
    "CnoNJ": ["--arm", "C", "--exclude-types", "gaussian_noise", "jpeg"],
    "BN": ["--arm", "B", "--generic-noise-jpeg"],
}
DEFAULT_QUEUE = [(a, s) for s in (1, 2, 3) for a in ("CnoNJ", "BN")]


def commands(arm, seed, weights, train_fraction):
    tag = f"cswin_{arm}_s{seed}"
    train = [PYTHON, "scripts/train_ablation.py", *ARMS[arm], "--seed", str(seed), "--name", tag,
             "--weights", weights, "--train-fraction", str(train_fraction)]
    last = os.path.join("experiments", "robust_runs", tag, "last.pth")
    evals = [PYTHON, "scripts/evaluate.py", "--checkpoint", last, "--tag", tag, "--model", "CSWin_64_24322_small_256",
             "--dedup-manifest", "data/manifests/author_data_val_dedup.csv"]
    rob = [PYTHON, "scripts/evaluate_robustness.py", "--checkpoint", last, "--tag", tag, "--model", "CSWin_64_24322_small_256",
           "--pool", "data/manifests/robust_pool.csv", "--dfdc", "data/manifests/dfdc_faces.csv"]
    return tag, train, evals, rob


def process(arm, seed, weights, train_fraction, dry):
    tag, train, evals, rob = commands(arm, seed, weights, train_fraction)
    ckpt_dir = os.path.join(ROOT, "experiments", "robust_runs", tag)
    pred_metrics = os.path.join(ROOT, "experiments", "predictions", f"{tag}_metrics.csv")
    rob_metrics = os.path.join(ROOT, "experiments", "robustness", f"{tag}_robust_metrics.csv")
    print("=" * 80)
    print(f"ABLATION RUN: arm {arm} | seed {seed} | tag {tag}")
    print("=" * 80)
    if dry:
        for c in (train, evals, rob):
            print("  " + " ".join(c))
        return
    rs.check_library_frozen()
    if rs.training_complete(ckpt_dir):
        print("Complete training found. Skipping training.")
    else:
        if os.path.isfile(os.path.join(ckpt_dir, "last.pth")):
            print("WARNING: interrupted run found (fewer epochs than expected). Retraining from scratch.")
        rs.run_cmd(train, f"Training {tag}")
    if os.path.isfile(pred_metrics):
        print(f"Standard metrics exist ({pred_metrics}). Skipping.")
    else:
        rs.run_cmd(evals, f"Evaluating standard splits for {tag}")
    if os.path.isfile(rob_metrics):
        print(f"Robustness metrics exist ({rob_metrics}). Skipping.")
    else:
        rs.run_cmd(rob, f"Evaluating robustness conditions for {tag}")
    print(f"[DONE] {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--seed", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--weights", default="checkpoints/cswin/cswin_small_224.pth")
    ap.add_argument("--train-fraction", type=float, default=0.5, help="must equal the value used for arms A, B, C")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.arm and args.seed:
        queue = [(args.arm, args.seed)]
    elif args.arm:
        queue = [(args.arm, s) for s in (1, 2, 3)]
    elif args.seed:
        queue = [(a, args.seed) for a in ARMS]
    else:
        queue = DEFAULT_QUEUE
    t0 = time.time()
    print(f"Ablation queue ({len(queue)} runs): {queue}")
    for arm, seed in queue:
        process(arm, seed, args.weights, args.train_fraction, args.dry_run)
    if not args.dry_run:
        print(f"ALL ABLATION RUNS COMPLETED in {(time.time() - t0) / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
