"""
shortcut_check.py

Audit: can the label be predicted from the corruption plan alone?

If corruption were applied more often (or more strongly) to one class, a detector could learn
"corrupted => fake" (or real) instead of learning the real task. This is exactly the shortcut
your clean-trained models already show for noise. The audit regenerates, WITHOUT touching any
image, the plan that training arm C would apply to every training sample in every epoch (it
calls the same StructuredCorruptor.plan_for used by the training dataset), then checks:

  1. corruption rate for real vs fake (difference and 95% CI),
  2. chi-square independence of label vs each corruption type,
  3. a logistic-regression "metadata classifier" that tries to predict the label from the plan
     (5-fold cross-validated AUC; should be ~0.50).

Run (no GPU, no images needed):
  python scripts/shortcut_check.py --manifest data/manifests/author_data.csv --epochs 3
  python scripts/shortcut_check.py --manifest data/manifests/author_data.csv --negative-control   # must FAIL

Index order = rows of the manifest with split == train, which is the order DeepfakeDataset uses.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "robustness")):
        break
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from robustness.corruptions import StructuredCorruptor, TRAIN_FAMILY  # noqa: E402


def plan_features(plan):
    f = np.zeros(2 + 2 * len(TRAIN_FAMILY), dtype=np.float32)
    f[0] = 1.0 if plan else 0.0
    f[1] = len(plan)
    for op in plan:
        k = TRAIN_FAMILY.index(op["type"])
        f[2 + k] = 1.0
        f[2 + len(TRAIN_FAMILY) + k] = op["severity"]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/author_data.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--max-ops", type=int, default=2)
    ap.add_argument("--negative-control", action="store_true",
                    help="deliberately corrupt reals 10 points more often; the audit must report FAIL")
    ap.add_argument("--out", default="experiments/robustness/shortcut_check.json")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man = man[man["split"] == args.split].reset_index(drop=True)
    labels_one = man["label"].values.astype(int)
    N = len(man)
    cor = StructuredCorruptor(p=args.p, family="T", max_ops=args.max_ops)
    cor_real = StructuredCorruptor(p=min(1.0, args.p + 0.10), family="T", max_ops=args.max_ops) if args.negative_control else cor
    print(f"samples per epoch: {N:,} | epochs: {args.epochs} | negative control: {args.negative_control}")

    X, y = [], []
    for e in range(1, args.epochs + 1):
        for i in range(N):
            c = cor_real if (args.negative_control and labels_one[i] == 1) else cor
            X.append(plan_features(c.plan_for(args.seed, e, i)))
            y.append(labels_one[i])
    X = np.asarray(X)
    y = np.asarray(y)

    # 1) corruption rate by class
    any_c = X[:, 0]
    p_real, p_fake = any_c[y == 1].mean(), any_c[y == 0].mean()
    n_r, n_f = int((y == 1).sum()), int((y == 0).sum())
    diff = p_real - p_fake
    se = np.sqrt(p_real * (1 - p_real) / n_r + p_fake * (1 - p_fake) / n_f)
    print(f"\ncorruption rate: real {p_real:.4f} | fake {p_fake:.4f} | diff {diff:+.4f} (95% CI {diff - 1.96 * se:+.4f} .. {diff + 1.96 * se:+.4f})")

    # 2) chi-square per indicator
    names = ["any_corruption"] + TRAIN_FAMILY
    cols = [0] + [2 + k for k in range(len(TRAIN_FAMILY))]
    pvals = {}
    for nm, c in zip(names, cols):
        table = pd.crosstab(y, X[:, c] > 0)
        if table.shape == (2, 2):
            pvals[nm] = float(chi2_contingency(table)[1])
        else:
            pvals[nm] = 1.0
    print("chi-square p-values (label vs indicator):", {k: round(v, 4) for k, v in pvals.items()})

    # 3) metadata classifier
    clf = LogisticRegression(max_iter=300)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y) // 4)
        if len(set(y[idx].tolist())) == 2:
            boots.append(roc_auc_score(y[idx], prob[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"metadata classifier AUC (5-fold CV): {auc:.4f} (bootstrap 95% CI {lo:.4f} .. {hi:.4f}); chance = 0.5000")

    passed = (abs(auc - 0.5) < 0.01) and (abs(diff) < 0.01) and all(v > 0.001 for v in pvals.values())
    verdict = "PASS" if passed else "FAIL"
    if args.negative_control:
        verdict += " (this is the NEGATIVE CONTROL: FAIL is the correct, expected outcome)" if not passed else \
            " (WARNING: the audit did not detect the deliberate leak; do not trust it)"
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"negative_control": args.negative_control, "n_samples": int(len(y)), "rate_real": float(p_real),
                   "rate_fake": float(p_fake), "rate_diff": float(diff), "chi2_p": pvals,
                   "metadata_auc": auc, "metadata_auc_ci": [float(lo), float(hi)], "verdict": verdict}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
