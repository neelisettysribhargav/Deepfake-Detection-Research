"""
inspect_consensus_misses.py

Which test REAL images do every model call fake, and what do they look like?

Reads the per-image prediction files written by eval_predictions.py, finds test reals that
ALL listed models classify as fake (prob < threshold) and test reals that ALL classify as real,
then compares them (and train / validation reals) on simple image statistics and writes
contact sheets so you can look at them.

Run from the project root (paths in the prediction files are relative to it):

  python inspect_consensus_misses.py ^
      --preds experiments/predictions/cswin_pre_a_predictions.csv ^
              experiments/predictions/convnext_tiny_pre_predictions.csv ^
              experiments/predictions/xception_pre_predictions.csv ^
      --out experiments/inspect

Outputs in --out:
  consensus_missed_reals.csv   path, subset, and each model's probability
  stats.csv                    per-group medians and quartiles of each statistic
  grid_missed.png              test reals every model calls fake
  grid_correct.png             test reals every model gets right
  grid_train.png               train reals
  grid_val.png                 validation reals
Descriptive only: nothing here trains, tunes or touches thresholds.
"""

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp


def read_preds(path):
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype(str).str.strip()
    return d


def image_stats(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(lap.var()),
        "saturation": float(hsv[..., 1].mean()),
        "file_kb": os.path.getsize(path) / 1024.0,
    }


def grid(paths, out_png, cols=8, thumb=128):
    rows = int(np.ceil(len(paths) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            continue
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/inspect")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-grid", type=int, default=48)
    ap.add_argument("--n-stats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = [read_preds(p) for p in args.preds]
    names = [os.path.basename(p).replace("_predictions.csv", "") for p in args.preds]

    test = [f[f.split == "test"].set_index("path") for f in frames]
    base = test[0]
    reals = base.index[base.label == 1]
    prob = pd.DataFrame({n: t.loc[reals, "prob"] for n, t in zip(names, test)})
    missed = prob.index[(prob < args.threshold).all(axis=1)]
    correct = prob.index[(prob >= args.threshold).all(axis=1)]
    print(f"test reals: {len(reals)} | missed by ALL models: {len(missed)} | correct for ALL models: {len(correct)}")
    print("missed by subset:", base.loc[missed, "subset"].value_counts().to_dict())

    out = prob.loc[missed].copy()
    out.insert(0, "subset", base.loc[missed, "subset"])
    out.to_csv(os.path.join(args.out, "consensus_missed_reals.csv"))

    f0 = frames[0]
    train_reals = f0[(f0.split == "train_sample") & (f0.label == 1)].path.tolist()
    val_reals = f0[(f0.split == "validation") & (f0.label == 1)].path.tolist()

    groups = {
        "test_missed": list(missed),
        "test_correct": list(correct),
        "train_reals": train_reals,
        "val_reals": val_reals,
    }

    stat_rows, per_group = [], {}
    for g, paths in groups.items():
        sel = rng.choice(len(paths), size=min(args.n_stats, len(paths)), replace=False)
        stats = pd.DataFrame([image_stats(paths[i]) for i in sel])
        per_group[g] = stats
        row = {"group": g, "n": len(stats)}
        for c in stats.columns:
            q1, med, q3 = stats[c].quantile([0.25, 0.5, 0.75])
            row[f"{c}_median"], row[f"{c}_q1"], row[f"{c}_q3"] = round(med, 2), round(q1, 2), round(q3, 2)
        stat_rows.append(row)
        gp = rng.choice(len(paths), size=min(args.n_grid, len(paths)), replace=False)
        grid([paths[i] for i in gp], os.path.join(args.out, f"grid_{g.split('_', 1)[1] if g.startswith('test_') else g.split('_')[0]}.png"))

    st = pd.DataFrame(stat_rows)
    st.to_csv(os.path.join(args.out, "stats.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nmedians (q1-q3 in stats.csv):")
    print(st[["group", "n"] + [c for c in st.columns if c.endswith("_median")]].to_string(index=False))

    print("\nKS tests, test_missed vs test_correct (p-values):")
    for c in per_group["test_missed"].columns:
        ks = ks_2samp(per_group["test_missed"][c], per_group["test_correct"][c])
        print(f"  {c:<11} KS={ks.statistic:.3f}  p={ks.pvalue:.2e}")
    print(f"\nContact sheets and CSVs saved in {args.out}")


if __name__ == "__main__":
    main()
