"""
train_arms.py

Trains one run of one arm. Everything except the augmentation is identical across arms:

  arm A  clean images only (control)
  arm B  generic augmentation: ColorJitter (p=.5), GaussianBlur (p=.3), RandomErasing (p=.25)
  arm C  structured corruption from family T, applied to p_corrupt of the images (default 0.5),
         same distribution for real and fake (the sampler never sees the label)

Same model init (official ImageNet CSWin-S), same optimiser/schedule/epochs/seed handling.
The checkpoint to evaluate is the LAST epoch, for every arm (no checkpoint picking on clean
validation, which would favour arm A). Per-epoch monitoring uses a fixed clean validation sample
and a fixed synthetic-corrupted (family T) validation sample.

Run from the project root, e.g. (about 3 h per run on the full train set):
  python scripts/train_arms.py --arm C --seed 1 --weights checkpoints/cswin/cswin_small_224.pth

Budget option: --train-fraction 0.5 draws a fresh random half every epoch (the whole dataset is
still covered across epochs). Use the SAME setting for all arms.
Second backbone: --arch convnext_tiny (no --weights needed).
"""

import os
import sys
import csv
import json
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _ in range(4):
    if os.path.isdir(os.path.join(ROOT, "models")) and os.path.isdir(os.path.join(ROOT, "datasets")):
        break
    ROOT = os.path.dirname(ROOT)
for _p in (os.path.join(ROOT, "scripts"), HERE, ROOT):      # ROOT first so the local `datasets` package wins
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from datasets import DeepfakeDataset, get_test_transform                    # noqa: E402
from datasets.transforms import FastResize, IMAGENET_MEAN, IMAGENET_STD     # noqa: E402
from models.model_factory import get_model                                  # noqa: E402
from robustness.corruptions import StructuredCorruptor                      # noqa: E402
from robustness.datasets_robust import CorruptedTrainDataset                # noqa: E402

SUBSETS = ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]
TRAIN_MANIFEST = "data/manifests/author_data.csv"
VAL_MANIFEST = "data/manifests/author_data_val_dedup.csv"
CSWIN_NAME = "CSWin_64_24322_small_256"
VAL_SAMPLE_SEED = 12345          # fixed, identical for every arm and seed


def build_model(args):
    if args.arch == "cswin_pretrained":
        if not args.weights:
            raise SystemExit("--weights is required for cswin_pretrained")
        try:
            from load_cswin import load_official_cswin_weights
        except ImportError:
            from scripts.load_cswin import load_official_cswin_weights
        model = get_model(CSWIN_NAME, num_classes=1, pretrained=False, drop_rate=0.0, block_drop_rate=0.0,
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
        buckets.setdefault((is_head, p.ndim <= 1), []).append(p)
    groups = []
    for (is_head, no_decay), params in buckets.items():
        base = lr * (head_mult if is_head else 1.0)
        groups.append({"params": params, "lr": base, "base_lr": base, "weight_decay": 0.0 if no_decay else wd})
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
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x.to(device, non_blocking=True)).float().view(-1)
        logits.append(out.cpu())
        ys.append(torch.as_tensor(y).float())
    logits, ys = torch.cat(logits), torch.cat(ys)
    prob = torch.sigmoid(logits).numpy()
    y = ys.numpy().astype(int)
    pred = (prob >= 0.5).astype(int)
    return {"acc": float((pred == y).mean()),
            "sn": float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
            "sp": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
            "auc": float(roc_auc_score(y, prob)) if len(set(y.tolist())) == 2 else float("nan")}


def make_transforms(arm):
    base = [FastResize(256), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if arm == "B":
        pre = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3)])
        post = transforms.Compose(base + [transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random")])
        return pre, post
    return None, transforms.Compose(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", default="cswin_pretrained", choices=["cswin_pretrained", "convnext_tiny"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--p-corrupt", type=float, default=0.5, help="arm C: fraction of images corrupted")
    ap.add_argument("--max-ops", type=int, default=2, help="arm C: max corruptions stacked on one image")
    ap.add_argument("--val-sample", type=int, default=8000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default="experiments/robust_runs")
    args = ap.parse_args()

    prefix = "cswin" if args.arch == "cswin_pretrained" else args.arch
    name = args.name or f"{prefix}_{args.arm}_s{args.seed}"
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pre, post = make_transforms(args.arm)
    eval_post = get_test_transform(256)
    corruptor = StructuredCorruptor(p=args.p_corrupt, family="T", max_ops=args.max_ops) if args.arm == "C" else None

    base_train = DeepfakeDataset(subsets=SUBSETS, split="train", transform=None, manifest_path=TRAIN_MANIFEST)
    train_ds = CorruptedTrainDataset(base_train, post, corruptor, pre_transform=pre, seed=args.seed)
    N = len(train_ds)
    n_epoch = int(N * args.train_fraction) if args.train_fraction < 1.0 else N
    steps_per_epoch = n_epoch // args.batch_size

    base_val = DeepfakeDataset(subsets=SUBSETS, split="validation", transform=None, manifest_path=VAL_MANIFEST)
    vrng = np.random.default_rng(VAL_SAMPLE_SEED)
    val_idx = sorted(vrng.choice(len(base_val), size=min(args.val_sample, len(base_val)), replace=False).tolist())
    val_clean = Subset(CorruptedTrainDataset(base_val, eval_post, None), val_idx)
    val_corr = Subset(CorruptedTrainDataset(base_val, eval_post, StructuredCorruptor(p=1.0, family="T"), seed=777), val_idx)
    mk = lambda d: DataLoader(d, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_clean_loader, val_corr_loader = mk(val_clean), mk(val_corr)
    print(f"[{name}] train {N:,} (per epoch {n_epoch:,}) | val sample {len(val_idx):,} | arm {args.arm} | seed {args.seed}")

    model = build_model(args).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)) if args.ema_decay > 0 else None
    optimizer = torch.optim.AdamW(make_param_groups(model, args.lr, args.head_lr_mult, args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    criterion = nn.BCEWithLogitsLoss()
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    step = 0

    log_path = os.path.join(out_dir, "history.csv")
    cols = ["epoch", "lr_end", "train_loss", "train_acc",
            "clean_acc", "clean_auc", "clean_sn", "clean_sp", "corrT_acc", "corrT_auc", "corrT_sn", "corrT_sp", "minutes"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cols)

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        if args.train_fraction < 1.0:
            idx = np.random.default_rng([args.seed, epoch, 99]).choice(N, size=n_epoch, replace=False).tolist()
            data = Subset(train_ds, idx)
        else:
            data = train_ds
        g = torch.Generator()
        g.manual_seed(args.seed * 1000 + epoch)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in tqdm(loader, ncols=85, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            f_ = lr_factor(step, warmup_steps, total_steps)
            for gr in optimizer.param_groups:
                gr["lr"] = gr["base_lr"] * f_
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
        mc = evaluate(eval_model, val_clean_loader, device, use_cuda)
        mr = evaluate(eval_model, val_corr_loader, device, use_cuda)
        minutes = (time.time() - t0) / 60.0
        lr_end = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch} | train acc {100 * correct / seen:.2f}% | clean val acc {100 * mc['acc']:.2f}% auc {mc['auc']:.4f} "
              f"| corrupted-T val acc {100 * mr['acc']:.2f}% auc {mr['auc']:.4f} | {minutes:.1f} min")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{lr_end:.3e}", f"{loss_sum / seen:.5f}", f"{100 * correct / seen:.3f}",
                                    f"{100 * mc['acc']:.3f}", f"{mc['auc']:.5f}", f"{100 * mc['sn']:.3f}", f"{100 * mc['sp']:.3f}",
                                    f"{100 * mr['acc']:.3f}", f"{mr['auc']:.5f}", f"{100 * mr['sn']:.3f}", f"{100 * mr['sp']:.3f}",
                                    f"{minutes:.1f}"])
        torch.save((ema.module if ema is not None else model).state_dict(), os.path.join(out_dir, "last.pth"))

    model_name = CSWIN_NAME if args.arch == "cswin_pretrained" else args.arch
    last = os.path.join(out_dir, "last.pth")
    print("\nDone. Evaluate the LAST epoch (pre-specified rule) with:")
    print(f"  python scripts/evaluate.py --checkpoint {last} --tag {name} --model {model_name} --dedup-manifest {VAL_MANIFEST}")
    print(f"  python scripts/evaluate_robustness.py --checkpoint {last} --tag {name} --model {model_name} "
          f"--pool data/manifests/robust_pool.csv --dfdc data/manifests/dfdc_faces.csv")


if __name__ == "__main__":
    main()
