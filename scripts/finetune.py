"""
finetune_pretrained.py

Fine-tunes an ImageNet-pretrained model on the frozen author_data manifest (full train split)
and selects the checkpoint on DEDUPLICATED validation. The test set is never touched here.

Architectures:
  cswin_pretrained   project CSWin-S initialised from the official ImageNet-1K checkpoint
                     (--weights path/to/cswin_small_224.pth; check it first with
                     load_pretrained_cswin.py)
  convnext_tiny | xception | inception_v3   pretrained models from get_model(..., pretrained=True)

Recipe (standard fine-tuning, no image augmentation): AdamW, backbone LR --lr, head LR x10,
per-iteration linear warmup + cosine decay, weight decay 0.05 (not on bias/norm), gradient
clipping 1.0, fp16 autocast, optional label smoothing and EMA of weights.

First pass to see whether it works (about half the cost of an epoch-equivalent each):
  python finetune_pretrained.py --arch cswin_pretrained --weights cswin_small_224.pth ^
      --epochs 3 --train-fraction 0.5 --name cswin_pre_a

Then evaluate the chosen checkpoint ONCE on test with eval_predictions.py (command is printed).
"""

import os
import sys
import csv
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.dirname(HERE))

from datasets import DeepfakeDataset, get_train_transform, get_test_transform
from models.model_factory import get_model

SUBSETS = ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]
TRAIN_MANIFEST = "data/manifests/author_data.csv"
VAL_MANIFEST = "data/manifests/author_data_val_dedup.csv"
CSWIN_NAME = "CSWin_64_24322_small_256"


def build_model(args):
    if args.arch == "cswin_pretrained":
        if not args.weights:
            raise SystemExit("--weights is required for cswin_pretrained")
        try:
            from load_cswin import load_official_cswin_weights
        except ImportError:
            from scripts.load_cswin import load_official_cswin_weights
        model = get_model(CSWIN_NAME, num_classes=1, pretrained=False,
                          drop_rate=0.0, block_drop_rate=0.0,
                          head_drop_rate=args.head_dropout, drop_path_rate=args.drop_path)
        load_official_cswin_weights(model, args.weights)
        return model
    return get_model(args.arch, num_classes=1, pretrained=True)


def make_param_groups(model, lr, head_mult, wd):
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith(("head.", "classifier.", "fc."))
        no_decay = p.ndim <= 1
        buckets.setdefault((is_head, no_decay), []).append(p)
    groups = []
    for (is_head, no_decay), params in buckets.items():
        base = lr * (head_mult if is_head else 1.0)
        groups.append({"params": params, "lr": base, "base_lr": base,
                       "weight_decay": 0.0 if no_decay else wd})
    return groups


def lr_factor(step, warmup_steps, total_steps, floor=0.01):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, loader, device, use_cuda):
    model.eval()
    logits, ys = [], []
    for x, y in tqdm(loader, ncols=85, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x).float().view(-1)
        logits.append(out.cpu())
        ys.append(torch.as_tensor(y).float())
    logits = torch.cat(logits)
    ys = torch.cat(ys)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, ys).item()
    prob = torch.sigmoid(logits).numpy()
    y = ys.numpy().astype(int)
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y).mean())
    sn = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    sp = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    auc = float(roc_auc_score(y, prob)) if len(set(y.tolist())) == 2 else float("nan")
    return loss, acc, sn, sp, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["cswin_pretrained", "convnext_tiny", "xception", "inception_v3"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone LR; head uses x10")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--val-sample", type=int, default=8000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/finetune")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = DeepfakeDataset(subsets=SUBSETS, split="train",
                               transform=get_train_transform(256), manifest_path=TRAIN_MANIFEST)
    val_ds = DeepfakeDataset(subsets=SUBSETS, split="validation",
                             transform=get_test_transform(256), manifest_path=VAL_MANIFEST)

    if args.train_fraction < 1.0:
        n = int(len(train_ds) * args.train_fraction)
        train_ds = Subset(train_ds, sorted(rng.choice(len(train_ds), size=n, replace=False).tolist()))
    n_val = min(args.val_sample, len(val_ds))
    val_sub = Subset(val_ds, sorted(rng.choice(len(val_ds), size=n_val, replace=False).tolist()))
    print(f"train images: {len(train_ds):,} | per-epoch val sample: {len(val_sub):,} "
          f"(of {len(val_ds):,} dedup val)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)) if args.ema_decay > 0 else None

    optimizer = torch.optim.AdamW(make_param_groups(model, args.lr, args.head_lr_mult, args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    criterion = nn.BCEWithLogitsLoss()

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    step = 0
    best_acc = -1.0

    log_path = os.path.join(out_dir, "history.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr_backbone_end", "train_loss", "train_acc",
                                    "val_loss", "val_acc", "val_sn", "val_sp", "val_auc", "minutes"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in tqdm(train_loader, ncols=85, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            f_ = lr_factor(step, warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * f_
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            target = y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x).float().view(-1)
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            step += 1

            loss_sum += loss.item() * len(y)
            correct += ((torch.sigmoid(out) >= 0.5).float() == y).sum().item()
            seen += len(y)

        eval_model = ema if ema is not None else model
        v_loss, v_acc, v_sn, v_sp, v_auc = evaluate(eval_model, val_loader, device, use_cuda)
        minutes = (time.time() - t0) / 60.0
        lr_end = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d} | lr {lr_end:.2e} | train acc {100 * correct / seen:.2f}% "
              f"loss {loss_sum / seen:.4f} | dedup-val acc {100 * v_acc:.2f}% "
              f"Sn {100 * v_sn:.2f} Sp {100 * v_sp:.2f} AUC {v_auc:.4f} | {minutes:.1f} min")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{lr_end:.3e}", f"{loss_sum / seen:.5f}",
                                    f"{100 * correct / seen:.3f}", f"{v_loss:.5f}", f"{100 * v_acc:.3f}",
                                    f"{100 * v_sn:.3f}", f"{100 * v_sp:.3f}", f"{v_auc:.5f}", f"{minutes:.1f}"])

        sd = (ema.module if ema is not None else model).state_dict()
        torch.save(sd, os.path.join(out_dir, "last.pth"))
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(sd, os.path.join(out_dir, "best.pth"))

    model_name = CSWIN_NAME if args.arch == "cswin_pretrained" else args.arch
    print("\nDone. Best dedup-val sample accuracy: %.2f%%" % (100 * best_acc))
    print("Evaluate ONCE on test with:")
    print(f"  python scripts/evaluate.py --checkpoint {os.path.join(out_dir, 'best.pth')} "
          f"--tag {args.name} --model {model_name} --dedup-manifest {VAL_MANIFEST}")


if __name__ == "__main__":
    main()
