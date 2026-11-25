"""
run_convnext.py

Second-backbone replication with ConvNeXt-Tiny (pretrained, via get_model): does the
"noise/JPEG coverage closes the gap" finding hold beyond CSWin?

Default queue: arms A (clean), BN (generic + noise/JPEG), C (structured), seeds 1 and 2,
interleaved by seed (A s1, BN s1, C s1, A s2, ...). 6 runs. Add --include-b for generic
augmentation (arm B) and --seeds 1 2 3 for a third seed.

Same recipe as the CSWin study (AdamW 5e-5 backbone, head x10, 3 epochs, half the train set per
epoch). The recipe is deliberately NOT tuned for ConvNeXt, so the comparison between arms stays fair.

Safety rules are the same as run_study.py: a run counts as finished only when history.csv has all
epochs, the frozen corruption library is hash-checked before every job, finished steps are skipped.

  python scripts/run_convnext.py --dry-run
  python scripts/run_convnext.py
  python scripts/run_convnext.py --arm C --seed 1

Needs scripts/run_study.py and scripts/train_ablation.py next to it. The first run downloads the
pretrained ConvNeXt weights (internet required once).

Afterwards:
  python scripts/check_prediction_files.py
  python scripts/aggregate_study.py --prefix convnext_tiny --arms A BN C --seeds 1 2 --out experiments/study_results_convnext
"""

import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_study as rs                                           # noqa: E402

PYTHON, ROOT = rs.PYTHON, rs.ROOT
ARCH = "convnext_tiny"
PREFIX = "convnext_tiny"

ARMS = {                                  # arm -> (script, extra args)
    "A": ("scripts/train_arms.py", ["--arm", "A"]),
    "B": ("scripts/train_arms.py", ["--arm", "B"]),
    "C": ("scripts/train_arms.py", ["--arm", "C"]),
    "BN": ("scripts/train_ablation.py", ["--arm", "B", "--generic-noise-jpeg"]),
}


def commands(arm, seed, train_fraction):
    tag = f"{PREFIX}_{arm}_s{seed}"
    script, extra = ARMS[arm]
    train = [PYTHON, script, *extra, "--arch", ARCH, "--seed", str(seed), "--name", tag,
             "--train-fraction", str(train_fraction)]
    last = os.path.join("experiments", "robust_runs", tag, "last.pth")
    evals = [PYTHON, "scripts/evaluate.py", "--checkpoint", last, "--tag", tag, "--model", ARCH,
             "--dedup-manifest", "data/manifests/author_data_val_dedup.csv"]
    rob = [PYTHON, "scripts/evaluate_robustness.py", "--checkpoint", last, "--tag", tag, "--model", ARCH,
           "--pool", "data/manifests/robust_pool.csv", "--dfdc", "data/manifests/dfdc_faces.csv"]
    return tag, train, evals, rob


def process(arm, seed, train_fraction, dry):
    tag, train, evals, rob = commands(arm, seed, train_fraction)
    ckpt_dir = os.path.join(ROOT, "experiments", "robust_runs", tag)
    pred_metrics = os.path.join(ROOT, "experiments", "predictions", f"{tag}_metrics.csv")
    rob_metrics = os.path.join(ROOT, "experiments", "robustness", f"{tag}_robust_metrics.csv")
    print("=" * 80)
    print(f"CONVNEXT RUN: arm {arm} | seed {seed} | tag {tag}")
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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--include-b", action="store_true", help="also run arm B (generic augmentation)")
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = ["A", "B", "BN", "C"] if args.include_b else ["A", "BN", "C"]
    if args.arm and args.seed:
        queue = [(args.arm, args.seed)]
    elif args.arm:
        queue = [(args.arm, s) for s in args.seeds]
    elif args.seed:
        queue = [(a, args.seed) for a in arms]
    else:
        queue = [(a, s) for s in args.seeds for a in arms]
    t0 = time.time()
    print(f"ConvNeXt queue ({len(queue)} runs): {queue}")
    for arm, seed in queue:
        process(arm, seed, args.train_fraction, args.dry_run)
    if not args.dry_run:
        print(f"ALL CONVNEXT RUNS COMPLETED in {(time.time() - t0) / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
