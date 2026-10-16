"""
scripts/evaluate.py

One evaluation path for every number you report.

Runs a single checkpoint over the frozen manifest (validation, test, and a fixed random
sample of train), saves PER-IMAGE predictions, and computes every metric from that file.
Nothing is hard-coded, so tables for validation, deduplicated validation and test always
come from the same weights, the same preprocessing and the same threshold.

Run from the project root, e.g.:

  python scripts/evaluate.py ^
      --checkpoint experiments/checkpoints/cswin_scratch_v3/best_e10_frozen.pth ^
      --tag e10 ^
      --dedup-manifest data/manifests/author_data_val_dedup.csv

Outputs (in --out-dir, default experiments/predictions):
  <tag>_predictions.csv   split, path, subset, label, logit, prob
  <tag>_metrics.csv       one row per group (split / subset / dedup-validation)

Notes:
  * fp32 by default (deterministic and no autocast surprises). Use --amp only to compare.
  * The model is built with get_model(); if you add model options (e.g. ReLU), pass them
    with --model-kwargs '{"act": "relu"}' (get_model must forward the kwargs).
  * label 1 = real (positive class: Sn), label 0 = fake (Sp), as in the paper.
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.dirname(HERE))

from datasets import DeepfakeDataset, get_test_transform
from models.model_factory import get_model

SUBSETS = ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]


def load_model(name, ckpt_path, device, model_kwargs):
    model = get_model(name, num_classes=1, pretrained=False, **model_kwargs).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, ds, indices, device, batch_size, num_workers, amp):
    data = ds if indices is None else Subset(ds, list(indices))
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out_all = []
    for batch in tqdm(loader, ncols=85):
        images = batch[0].to(device, non_blocking=True)
        if amp:
            with torch.autocast("cuda"):
                out = model(images)
        else:
            out = model(images)
        out_all.append(out.float().view(-1).cpu())
    return torch.cat(out_all).numpy().astype(np.float64)


def add_rows(rows, split_name, ds, indices, logits):
    idx = range(len(ds)) if indices is None else indices
    for k, i in enumerate(idx):
        path, label, subset = ds.samples[i]
        rows.append((split_name, path, subset, int(label), float(logits[k])))


def metrics(y, prob, thr=0.5):
    y = np.asarray(y)
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)

    def div(a, b):
        return a / b if b else float("nan")

    sn, sp, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    f1 = div(2 * prec * sn, prec + sn) if (prec == prec and sn == sn) else float("nan")
    auc = roc_auc_score(y, prob) if len(set(y.tolist())) == 2 else float("nan")
    return {"n": n, "acc": (tp + tn) / n, "sn": sn, "sp": sp, "precision": prec,
            "f1": f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def read_dedup_val_paths(dedup_manifest):
    keep = set()
    with open(dedup_manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "validation":
                keep.add(row["path"])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="short name for this checkpoint, e.g. e10")
    ap.add_argument("--model", default="CSWin_64_24322_small_256")
    ap.add_argument("--model-kwargs", default="{}")
    ap.add_argument("--manifest", default="data/manifests/author_data.csv")
    ap.add_argument("--dedup-manifest", default=None)
    ap.add_argument("--out-dir", default="experiments/predictions")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--train-sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    model = load_model(args.model, args.checkpoint, device, json.loads(args.model_kwargs))
    tf = get_test_transform(256)
    rows = []

    for split in ["validation", "test"]:
        print(f"\n[{split}]")
        ds = DeepfakeDataset(subsets=SUBSETS, split=split, transform=tf, manifest_path=args.manifest)
        logits = predict_logits(model, ds, None, device, args.batch_size, args.num_workers, args.amp)
        add_rows(rows, split, ds, None, logits)

    if args.train_sample > 0:
        print("\n[train sample]")
        ds = DeepfakeDataset(subsets=SUBSETS, split="train", transform=tf, manifest_path=args.manifest)
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(ds), size=min(args.train_sample, len(ds)), replace=False).tolist())
        logits = predict_logits(model, ds, idx, device, args.batch_size, args.num_workers, args.amp)
        add_rows(rows, "train_sample", ds, idx, logits)

    # ---- save per-image predictions
    pred_path = os.path.join(args.out_dir, f"{args.tag}_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "subset", "label", "logit", "prob"])
        for split, path, subset, label, logit in rows:
            w.writerow([split, path, subset, label, f"{logit:.6f}", f"{1.0 / (1.0 + np.exp(-logit)):.8f}"])
    print(f"\nSaved per-image predictions: {pred_path}")

    # ---- metrics, all from the rows above
    split_arr = np.array([r[0] for r in rows])
    path_arr = np.array([r[1] for r in rows])
    subset_arr = np.array([r[2] for r in rows])
    label_arr = np.array([r[3] for r in rows])
    prob_arr = 1.0 / (1.0 + np.exp(-np.array([r[4] for r in rows])))

    groups = []
    for split in ["train_sample", "validation", "test"]:
        m = split_arr == split
        if m.sum() == 0:
            continue
        groups.append((f"{split}/all", m))
        for s in SUBSETS:
            ms = m & (subset_arr == s)
            if ms.sum() > 0:
                groups.append((f"{split}/{s}", ms))
    if args.dedup_manifest:
        keep = read_dedup_val_paths(args.dedup_manifest)
        in_keep = np.array([p in keep for p in path_arr])
        groups.append(("validation/dedup", (split_arr == "validation") & in_keep))

    out_rows = []
    print(f"\n{'group':<26}{'n':>7}{'acc':>8}{'Sn':>8}{'Sp':>8}{'prec':>8}{'F1':>8}{'AUC':>8}")
    for name, m in groups:
        r = metrics(label_arr[m], prob_arr[m])
        out_rows.append({"group": name, **r})
        print(f"{name:<26}{r['n']:>7}{r['acc']*100:>8.2f}{r['sn']*100:>8.2f}{r['sp']*100:>8.2f}"
              f"{r['precision']*100:>8.2f}{r['f1']*100:>8.2f}{r['auc']:>8.4f}")

    met_path = os.path.join(args.out_dir, f"{args.tag}_metrics.csv")
    with open(met_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Saved metrics: {met_path}")


if __name__ == "__main__":
    main()
