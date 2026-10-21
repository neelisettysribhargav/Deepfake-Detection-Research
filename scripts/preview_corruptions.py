"""
preview_corruptions.py

Visual QA of the corruption library: one row per corruption type, one column per severity,
original image in the first column. LOOK at the output before running any experiment.

Run from the project root:
  python scripts/preview_corruptions.py --image "data/Dataset-1/Data Set 1/train/real/real_00001.jpg" ^
      --out experiments/robustness/preview.png

  --severities 1 3 5   (default)   --thumb 160   --seed 0
Family T rows (used in training) are listed first, then family H (held out, evaluation only).
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "robustness")):
        break
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from robustness.corruptions import TRAIN_FAMILY, HELDOUT_FAMILY, apply_corruption  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True, help="one or more image paths (one block per image)")
    ap.add_argument("--out", default="experiments/robustness/preview.png")
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    types = TRAIN_FAMILY + HELDOUT_FAMILY
    th = args.thumb
    label_w = 150
    cols = 1 + len(args.severities)
    block_h = th * len(types)
    W = label_w + cols * th * len(args.image)
    H = block_h
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)

    for r, t in enumerate(types):
        tag = ("[T] " if t in TRAIN_FAMILY else "[H] ") + t
        d.text((6, r * th + th // 2 - 5), tag, fill=(0, 0, 0))

    for k, path in enumerate(args.image):
        img = Image.open(path).convert("RGB").resize((256, 256))
        x_off = label_w + k * cols * th
        for r, t in enumerate(types):
            sheet.paste(img.resize((th, th)), (x_off, r * th))
            for c, s in enumerate(args.severities):
                out = apply_corruption(img, t, s, np.random.default_rng(args.seed + 1000 * r + s))
                sheet.paste(out.resize((th, th)), (x_off + (c + 1) * th, r * th))
                d.text((x_off + (c + 1) * th + 3, r * th + 3), f"s{s}", fill=(255, 255, 0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    print(f"saved {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
