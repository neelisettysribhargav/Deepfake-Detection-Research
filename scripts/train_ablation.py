"""
train_ablation.py

Runs scripts/train_arms.py with two ablation switches, WITHOUT editing train_arms.py or the
frozen corruption library (it patches names inside train_arms at run time).

  --exclude-types t1 t2 ...   (use with --arm C)
        remove corruption types from arm C's training sampler, e.g.
        --exclude-types gaussian_noise jpeg   ->  arm "C minus noise/JPEG"
  --generic-noise-jpeg        (use with --arm B)
        arm B's generic augmentation PLUS Gaussian noise and JPEG, applied with the same
        marginal frequency (about 10.8% of images each) and the same severity tables as in
        arm C. This is the coverage-matched generic baseline, "B plus noise/JPEG".

--name is REQUIRED with either switch, so results never collide with the main arms
(use names like cswin_CnoNJ_s1 and cswin_BN_s1; the aggregation script finds them as arms
"CnoNJ" and "BN"). All other arguments are passed to train_arms.py unchanged.

  python scripts/train_ablation.py --arm C --exclude-types gaussian_noise jpeg --seed 1 ^
      --name cswin_CnoNJ_s1 --weights checkpoints/cswin/cswin_small_224.pth --train-fraction 0.5
  python scripts/train_ablation.py --arm B --generic-noise-jpeg --seed 1 ^
      --name cswin_BN_s1 --weights checkpoints/cswin/cswin_small_224.pth --train-fraction 0.5
"""

import os
import sys
import random
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "robustness")):
        break
    ROOT = os.path.dirname(ROOT)
for _p in (os.path.join(ROOT, "scripts"), HERE, ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from robustness.corruptions import TRAIN_FAMILY, apply_corruption   # noqa: E402

# marginal frequency of one specific type in arm C with defaults p=0.5, max_ops=2, two_op_prob=0.3:
# 0.5 * (0.7 * 1/6 + 0.3 * 2/6) = 0.1083
DEFAULT_MARGINAL = 0.5 * (0.7 * (1 / 6) + 0.3 * (2 / 6))


class NoiseJpeg:
    """PIL-level transform: Gaussian noise then JPEG, each independently with probability p.
    Uses python's `random`, which PyTorch DataLoader seeds differently in every worker."""

    def __init__(self, p_noise, p_jpeg):
        self.p = (("gaussian_noise", p_noise), ("jpeg", p_jpeg))

    def __call__(self, img):
        for ctype, p in self.p:
            if random.random() < p:
                img = apply_corruption(img, ctype, random.randint(1, 5), np.random.default_rng(random.randrange(2 ** 31 - 1)))
        return img


def arm_in(argv):
    return argv[argv.index("--arm") + 1] if "--arm" in argv else None


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--exclude-types", nargs="*", default=[])
    ap.add_argument("--generic-noise-jpeg", action="store_true")
    ap.add_argument("--bn-prob", type=float, default=DEFAULT_MARGINAL)
    ap.add_argument("--name", default=None)
    known, rest = ap.parse_known_args()
    arm = arm_in(rest)

    if not (known.exclude_types or known.generic_noise_jpeg):
        raise SystemExit("nothing to do: pass --exclude-types and/or --generic-noise-jpeg (otherwise use train_arms.py)")
    if not known.name:
        raise SystemExit("--name is required for ablation runs (e.g. cswin_CnoNJ_s1)")
    if known.exclude_types and arm != "C":
        raise SystemExit("--exclude-types only makes sense with --arm C")
    if known.generic_noise_jpeg and arm != "B":
        raise SystemExit("--generic-noise-jpeg only makes sense with --arm B")
    bad = [t for t in known.exclude_types if t not in TRAIN_FAMILY]
    if bad:
        raise SystemExit(f"unknown training types {bad}; choose from {TRAIN_FAMILY}")
    if len(set(known.exclude_types)) >= len(TRAIN_FAMILY):
        raise SystemExit("cannot exclude every corruption type")

    sys.argv = [sys.argv[0]] + rest + ["--name", known.name]
    import train_arms

    if known.exclude_types:
        excl = set(known.exclude_types)
        Base = train_arms.StructuredCorruptor

        class FilteredCorruptor(Base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.types = [t for t in self.types if t not in excl]

        train_arms.StructuredCorruptor = FilteredCorruptor
        print(f"[ablation] arm C sampler excludes {sorted(excl)}; remaining types: "
              f"{[t for t in TRAIN_FAMILY if t not in excl]}")

    if known.generic_noise_jpeg:
        orig = train_arms.make_transforms

        def patched(a):
            pre, post = orig(a)
            if a != "B" or pre is None:
                return pre, post
            extra = NoiseJpeg(known.bn_prob, known.bn_prob)
            return (lambda img: extra(pre(img))), post

        train_arms.make_transforms = patched
        print(f"[ablation] arm B gets Gaussian noise and JPEG with probability {known.bn_prob:.4f} each")

    train_arms.main()


if __name__ == "__main__":
    main()
