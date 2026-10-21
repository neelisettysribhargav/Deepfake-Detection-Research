"""
make_robust_pool.py

Builds the FROZEN pool of base images for synthetic-corruption evaluation:
  * source: the official TEST split of author_data.csv (held-out sources, not seen in training)
  * filter: images with natural noise (Immerkaer sigma) <= --max-sigma, using the features file
            written by corruption_prevalence.py / measure_noise.py. This keeps already-noisy
            test images out of the pool, so a synthetic corruption is applied to an
            (approximately) clean image. Other natural corruptions (colour casts, pixel blocks)
            are NOT detected by this filter; say so in the paper.
  * sample: --n-per-class real and --n-per-class fake, fixed seed.

The same pool is used by every model and every corruption condition (paired design).
Do not regenerate it after you start running experiments.

  python scripts/make_robust_pool.py --manifest data/manifests/author_data.csv ^
      --features experiments/inspect/image_features.csv --out data/manifests/robust_pool.csv
"""

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/author_data.csv")
    ap.add_argument("--features", default="experiments/inspect/image_features.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-sigma", type=float, default=3.0)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/manifests/robust_pool.csv")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man = man[man["split"] == args.split].copy()
    feat = pd.read_csv(args.features, skipinitialspace=True)
    feat.columns = [c.strip() for c in feat.columns]
    feat = feat[["path", "noise_sigma"]].drop_duplicates("path")

    df = man.merge(feat, on="path", how="left")
    n_all = len(df)
    n_missing = int(df.noise_sigma.isna().sum())
    df = df.dropna(subset=["noise_sigma"])
    kept = df[df.noise_sigma <= args.max_sigma]
    print(f"{args.split} images: {n_all} | without features: {n_missing} | "
          f"noise_sigma <= {args.max_sigma}: {len(kept)} ({100 * len(kept) / max(1, len(df)):.1f}% of those with features)")

    parts = []
    for label in (0, 1):
        pool = kept[kept.label == label]
        if len(pool) < args.n_per_class:
            raise SystemExit(f"only {len(pool)} images with label {label}; lower --n-per-class")
        parts.append(pool.sample(args.n_per_class, random_state=args.seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"saved {args.out}: {len(out)} images")
    print(out.groupby(["subset", "class_name"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
