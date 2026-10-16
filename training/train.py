import sys
import os
import time
import warnings
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DeepfakeDataset, get_train_transform, get_test_transform
from models.model_factory import get_model
from training.validate import validate
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import save_history

set_seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def build_optimizer(model, config):
    opt_type = config["training"].get("optimizer", "AdamW").lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.05))

    if opt_type == "adamw":
        print(f"Using AdamW Optimizer (lr={lr}, weight_decay={weight_decay})")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"Using Adam Optimizer (lr={lr}, weight_decay={weight_decay})")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = lr

    return optimizer

def build_scheduler(optimizer, config, total_epochs, start_epoch=0):
    scheduler_type = config["training"].get("scheduler", "cosine")
    warmup_epochs = config["training"].get("warmup_epochs", 0)

    if scheduler_type == "cosine":
        if warmup_epochs > 0 and warmup_epochs < total_epochs:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(total_epochs - warmup_epochs),
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for _ in range(start_epoch):
                    scheduler.step()
            return scheduler
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for _ in range(start_epoch):
                    scheduler.step()
            return scheduler
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')) else "configs/cswin_baseline.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    exp_name = config["experiment"]["name"]
    checkpoint_dir = f"{config['paths']['checkpoint_dir']}/{exp_name}"
    log_path = f"{config['paths']['log_dir']}/{exp_name}_history.csv"

    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    # Dataset & Preprocessing (Section 2.1: 256x256 image size, no augmentation)
    image_size = config["data"]["image_size"]

    manifest_path = config["data"].get("manifest", None)

    train_dataset = DeepfakeDataset(
        root_dir=config["data"].get("root"),
        subsets=config["data"]["train_subsets"],
        split="train",
        transform=get_train_transform(image_size),
        manifest_path=manifest_path
    )

    val_dataset = DeepfakeDataset(
        root_dir=config["data"].get("root"),
        subsets=config["data"]["val_subsets"],
        split="validation",
        transform=get_test_transform(image_size),
        manifest_path=manifest_path
    )

    print(f"Train samples loaded: {len(train_dataset)}")
    print(f"Validation samples loaded: {len(val_dataset)}")

    # Dataloaders
    batch_size = config["training"]["batch_size"]
    val_batch_size = config["training"].get("val_batch_size", batch_size * 2)
    num_workers = config["training"]["num_workers"]

    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor
    )

    # Model instantiation with all forwarded YAML options
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    num_classes = model_cfg.pop("num_classes", 1)
    pretrained = model_cfg.pop("pretrained", False)
    checkpoint_path = model_cfg.pop("checkpoint", None)

    model = get_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        checkpoint=checkpoint_path,
        **model_cfg
    )
    model = model.to(device)

    # Loss function (Section 3.1: Binary Cross-Entropy Loss)
    if num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using Binary Cross-Entropy Loss (BCEWithLogitsLoss)")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using Multi-Class Cross-Entropy Loss")

    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    best_f1 = 0.0
    total_epochs = config["training"]["epochs"]

    # Resuming checkpoint
    resume_cfg = config.get("resume", {})

    if resume_cfg.get("enabled", False) and os.path.exists(resume_cfg.get("checkpoint", "")):
        ckpt_path = resume_cfg["checkpoint"]
        weights_only = resume_cfg.get("weights_only", False)

        if weights_only:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_f1 = float(ckpt.get("best_f1", ckpt.get("metric", 0.0)))

            base_lr = float(config["training"]["learning_rate"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
                param_group["initial_lr"] = base_lr

            scheduler = build_scheduler(optimizer, config, total_epochs, start_epoch=start_epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Resumed weights only from {ckpt_path} (epoch {start_epoch}) with fresh optimizer (Learning Rate: {current_lr:.6f})")
        else:
            last_epoch, best_f1 = load_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler=None,  # Re-initialize scheduler with correct total_epochs
                scaler=scaler,
                device=device
            )
            start_epoch = last_epoch

            base_lr = config["training"]["learning_rate"]
            for param_group in optimizer.param_groups:
                param_group["initial_lr"] = base_lr

            scheduler = build_scheduler(optimizer, config, total_epochs, start_epoch=start_epoch)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Resumed training from epoch {start_epoch + 1}/{total_epochs} (Learning Rate: {current_lr:.6f}, Best F1: {best_f1:.4f})")
    else:
        scheduler = build_scheduler(optimizer, config, total_epochs, start_epoch=0)
        warmup_ep = config["training"].get("warmup_epochs", 0)
        print(f"Starting training from scratch ({total_epochs} epochs, {warmup_ep} warmup epochs)")

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()

        running_loss = torch.tensor(0.0, device=device)
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            total=len(train_loader),
            ncols=100,
            bar_format="{desc:<12} {percentage:3.0f}%|{bar:16}| {n_fmt:>4}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            mininterval=0.5,
            leave=True,
            file=sys.stdout
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                if num_classes == 1:
                    targets_tensor = labels_device.float().view_as(outputs)
                    loss = criterion(outputs, targets_tensor)
                else:
                    targets_tensor = labels_device.long()
                    loss = criterion(outputs, targets_tensor)

            scaler.scale(loss).backward()

            grad_clip = config["training"].get("grad_clip", 1.0)
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                if num_classes == 1:
                    preds = (outputs.detach() > 0.0).long().view(-1)
                else:
                    preds = outputs.detach().argmax(dim=1)
                running_correct += (preds == labels_device).sum().item()
                running_total += labels_device.numel()

            running_loss += loss.detach()

            if batch_idx % 20 == 0 or (batch_idx + 1) == len(train_loader):
                cur_avg_loss = (running_loss / (batch_idx + 1)).item()
                cur_acc = (running_correct / max(1, running_total)) * 100
                progress_bar.set_postfix(
                    loss=f"{cur_avg_loss:.4f}",
                    acc=f"{cur_acc:.1f}%",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )

        progress_bar.close()
        train_loss = (running_loss / len(train_loader)).item() if len(train_loader) > 0 else 0.0
        train_acc = running_correct / max(1, running_total)

        # Validation on holdout validation set
        metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
        print(f"Train Loss : {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {metrics['loss']:.4f}")
        print(f"Accuracy   : {metrics['accuracy'] * 100:.2f}%")
        print(f"Sensitivity (Sn - Real): {metrics['sensitivity'] * 100:.2f}%")
        print(f"Specificity (Sp - Fake): {metrics['specificity'] * 100:.2f}%")
        print(f"Precision  : {metrics['precision'] * 100:.2f}%")
        print(f"F1-Score   : {metrics['f1'] * 100:.2f}%")
        print(f"Next Epoch LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save Best Checkpoint based on F1-Score
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            print(f"New best F1 score: {best_f1 * 100:.2f}% -> Saving best checkpoint.")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics["f1"],
                best_f1,
                f"{checkpoint_dir}/best.pth"
            )

        # Save Last Checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics["f1"],
            best_f1,
            f"{checkpoint_dir}/last.pth"
        )

        peak_memory = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Peak VRAM  : {peak_memory:.2f} MB")

        epoch_time = (time.time() - start_time) / 60
        print(f"Epoch Time : {epoch_time:.2f} minutes")

        # Periodic snapshot saving if save_every > 0
        save_every = int(config["training"].get("save_every", 0))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1},
                f"{checkpoint_dir}/weights_epoch_{epoch + 1:03d}.pth"
            )

        # Save History to CSV
        history = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "peak_vram_mb": peak_memory,
            "epoch_time_minutes": epoch_time
        }
        save_history(history, log_path)

if __name__ == "__main__":
    main()
    