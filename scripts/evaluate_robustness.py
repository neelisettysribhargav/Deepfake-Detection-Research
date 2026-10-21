"""
evaluate_robustness.py

Robustness profile of one checkpoint on the frozen pool (data/manifests/robust_pool.csv):
  * clean pool
  * family T conditions (training family) and family H conditions (held out), each at the
    chosen severities, applied with seeds that do not depend on the model (paired design)
  * optional cross-dataset condition (DFDC frames) with --dfdc

Writes per-image logits and one metrics row per condition (acc, balanced acc, AUC, Sn, Sp, ECE,
Brier). Sensitivity (real recall) and specificity (fake recall) are reported separately because
the failure you diagnosed is asymmetric. Real = label 1 = positive class.

  python scripts/evaluate_robustness.py --checkpoint experiments/finetune/cswin_pre_a/best.pth ^
      --tag cswin_pre_a --model CSWin_64_24322_small_256 ^
      --pool data/manifests/robust_pool.csv --dfdc data/manifests/dfdc_faces.csv

About 15-20 min per checkpoint with 3 severities (34 conditions x 2000 images, fp32).
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "models")) and os.path.isdir(os.path.join(ROOT, "datasets")):
        break
    ROOT = os.path.dirname(ROOT)
for _p in (os.path.join(ROOT, "scripts"), HERE, ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from datasets import DeepfakeDataset, get_test_transform                  # noqa: E402
from models.model_factory import get_model                                # noqa: E402
from robustness.corruptions import TRAIN_FAMILY, HELDOUT_FAMILY, family_of  # noqa: E402
from robustness.datasets_robust import ConditionDataset                   # noqa: E402


def load_model(name, ckpt_path, device, model_kwargs):
    model = get_model(name, num_classes=1, pretrained=False, **model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, ds, device, batch_size, num_workers, amp):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    out_all = []
    for x, _ in tqdm(loader, ncols=85, leave=False):
        x = x.to(device, non_blocking=True)
        if amp:
            with torch.autocast("cuda"):
                o = model(x)
        else:
            o = model(x)
        out_all.append(o.float().view(-1).cpu())
    return torch.cat(out_all).numpy().astype(np.float64)


def calib(prob, y, n_bins=15):
    """ECE on predicted-class confidence, and Brier score on P(real)."""
    pred = (prob >= 0.5).astype(int)
    conf = np.maximum(prob, 1.0 - prob)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece), float(np.mean((prob - y) ** 2))


def metrics(y, prob):
    y = np.asarray(y).astype(int)
    pred = (prob >= 0.5).astype(int)
    sn = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    sp = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    ece, brier = calib(prob, y)
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "bal_acc": (sn + sp) / 2.0,
            "auc": float(roc_auc_score(y, prob)) if len(set(y.tolist())) == 2 else float("nan"),
            "sn": sn, "sp": sp, "ece": ece, "brier": brier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="CSWin_64_24322_small_256")
    ap.add_argument("--model-kwargs", default="{}")
    ap.add_argument("--pool", default="data/manifests/robust_pool.csv")
    ap.add_argument("--dfdc", default=None)
    ap.add_argument("--families", nargs="+", default=["T", "H"], choices=["T", "H"])
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--seed", type=int, default=2024, help="evaluation-corruption seed (keep fixed for every model)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out-dir", default="experiments/robustness")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model = load_model(args.model, args.checkpoint, device, json.loads(args.model_kwargs))
    post = get_test_transform(256)

    pool = DeepfakeDataset(manifest_path=args.pool, subsets=None, split=None, transform=None)
    jobs = [("pool", "clean", "clean", 0, pool)]
    for fam in args.families:
        for t in (TRAIN_FAMILY if fam == "T" else HELDOUT_FAMILY):
            for s in args.severities:
                jobs.append((f"{t}@{s}", family_of(t), t, s, pool))
    if args.dfdc:
        dfdc = DeepfakeDataset(manifest_path=args.dfdc, subsets=None, split=None, transform=None)
        jobs.append(("dfdc_clean", "DFDC", "clean", 0, dfdc))
    print(f"{len(jobs)} conditions | pool size {len(pool)}")

    pred_path = os.path.join(args.out_dir, f"{args.tag}_robust_predictions.csv")
    rows = []
    with open(pred_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["condition", "family", "type", "severity", "path", "label", "logit"])
        for cond, fam, ctype, sev, base in jobs:
            ds = ConditionDataset(base, post, None if ctype == "clean" else ctype, sev, seed=args.seed)
            print(f"-> {cond}")
            logits = predict_logits(model, ds, device, args.batch_size, args.num_workers, args.amp)
            y = np.array([base.samples[i][1] for i in range(len(base))])
            for i in range(len(base)):
                w.writerow([cond, fam, ctype, sev, base.samples[i][0], int(y[i]), f"{logits[i]:.5f}"])
            m = metrics(y, 1.0 / (1.0 + np.exp(-logits)))
            rows.append({"condition": cond, "family": fam, "type": ctype, "severity": sev, **m})
            print(f"   acc {100 * m['acc']:.2f} bal {100 * m['bal_acc']:.2f} auc {m['auc']:.4f} "
                  f"Sn {100 * m['sn']:.2f} Sp {100 * m['sp']:.2f} ece {m['ece']:.3f}")

    met_path = os.path.join(args.out_dir, f"{args.tag}_robust_metrics.csv")
    with open(met_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\nmean over corruption types, by family and severity (AUC | Sn | Sp, %):")
    for fam in ("T", "H"):
        for s in args.severities:
            sel = [r for r in rows if r["family"] == fam and r["severity"] == s]
            if sel:
                print(f"  family {fam} severity {s}: AUC {np.mean([r['auc'] for r in sel]):.4f} | "
                      f"Sn {100 * np.mean([r['sn'] for r in sel]):.1f} | Sp {100 * np.mean([r['sp'] for r in sel]):.1f}")
    print(f"\nSaved {met_path} and {pred_path}")


if __name__ == "__main__":
    main()
