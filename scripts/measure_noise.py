"""
corruption_prevalence.py

How common are noisy / corrupted-looking images in each split, subset and class?

The contact sheet of test reals that every model calls fake shows heavy speckle noise, colour
distortions and pixel blocks that are absent from the train and validation sheets. This script
measures that with numbers, for ALL classes and splits, so you can see:
  * how much of the test set looks corrupted, for real AND fake images,
  * whether train / validation contain any such images at all.

Per image it records a robust noise estimate (Immerkaer sigma), Laplacian variance, JPEG file
size, brightness and saturation. Purely descriptive; nothing is trained or tuned.

Run from the project root:
  python corruption_prevalence.py --manifest data/manifests/author_data.csv --out experiments/inspect

Options: --max-per-group N  (default 3000; use 0 for every image, slower)
Outputs: image_features.csv, prevalence_summary.csv
"""

import os
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def features(path):
    try:
        img = Image.open(path).convert("RGB")
        a = np.asarray(img, dtype=np.float32)
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        m = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
             - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
             + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
        sigma = float(np.sqrt(np.pi / 2.0) * np.abs(m).mean() / 6.0)
        lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        sat = float(np.asarray(img.convert("HSV"), dtype=np.float32)[..., 1].mean())
        return {"noise_sigma": sigma, "sharpness": float(lap.var()), "brightness": float(g.mean()),
                "saturation": sat, "file_kb": os.path.getsize(path) / 1024.0}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/author_data.csv")
    ap.add_argument("--out", default="experiments/inspect")
    ap.add_argument("--max-per-group", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = pd.read_csv(args.manifest, usecols=["path", "subset", "split", "class_name"])
    if args.max_per_group > 0:
        man = (man.groupby(["split", "subset", "class_name"], group_keys=False)
                  .apply(lambda d: d.sample(min(len(d), args.max_per_group), random_state=args.seed)))
    print(f"images to measure: {len(man):,}")

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        feats = list(tqdm(ex.map(features, man.path.tolist()), total=len(man), ncols=85))
    man = man.reset_index(drop=True)
    f = pd.DataFrame([x if x is not None else {} for x in feats])
    df = pd.concat([man, f], axis=1).dropna(subset=["noise_sigma"])
    df.to_csv(os.path.join(args.out, "image_features.csv"), index=False)

    thresholds = [6, 10, 15]
    rows = []
    for (split, subset, cls), d in df.groupby(["split", "subset", "class_name"]):
        row = {"split": split, "subset": subset, "class": cls, "n": len(d),
               "noise_median": round(d.noise_sigma.median(), 2),
               "noise_p90": round(d.noise_sigma.quantile(0.9), 2),
               "kb_median": round(d.file_kb.median(), 2)}
        for t in thresholds:
            row[f"%noise>{t}"] = round(100 * (d.noise_sigma > t).mean(), 1)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["split", "subset", "class"])
    summary.to_csv(os.path.join(args.out, "prevalence_summary.csv"), index=False)

    pd.set_option("display.width", 250)
    print("\nPer split / subset / class:")
    print(summary.to_string(index=False))

    print("\nPer split / class (all subsets):")
    rows2 = []
    for (split, cls), d in df.groupby(["split", "class_name"]):
        r = {"split": split, "class": cls, "n": len(d), "noise_median": round(d.noise_sigma.median(), 2)}
        for t in thresholds:
            r[f"%noise>{t}"] = round(100 * (d.noise_sigma > t).mean(), 1)
        rows2.append(r)
    print(pd.DataFrame(rows2).to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
