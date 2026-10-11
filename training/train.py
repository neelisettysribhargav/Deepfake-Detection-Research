import sys
import os
import time
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DeepfakeDataset, get_train_transform, get_test_transform
from models.model_factory import get_model
from training.validate import validate
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import save_history

torch.backends.cudnn.benchmark = True
set_seed(42)

def main():
    config_path = "configs/cswin_baseline.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    exp_name = config["experiment"]["name"]
    checkpoint_dir = f"{config['paths']['checkpoint_dir']}/{exp_name}"
    log_path = f"{config['paths']['log_dir']}/{exp_name}_history.csv"

    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    # Dataset & Preprocessing (Section 2.1: 256x256 image size, no augmentation)
    image_size = config["data"]["image_size"]

    train_dataset = DeepfakeDataset(
        root_dir=config["data"]["root"],
        subsets=config["data"]["train_subsets"],
        split="train",
        transform=get_train_transform(image_size)
    )

    val_dataset = DeepfakeDataset(
        root_dir=config["data"]["root"],
        subsets=config["data"]["val_subsets"],
        split="validation",
        transform=get_test_transform(image_size)
    )

    print(f"Train samples loaded: {len(train_dataset)}")
    print(f"Validation samples loaded: {len(val_dataset)}")

    # Dataloaders
    batch_size = config["training"]["batch_size"]
    val_batch_size = config["training"].get("val_batch_size", batch_size * 2)
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    # Model instantiation (CSWin-Small: 35M parameters, 256x256 resolution)
    num_classes = config["model"]["num_classes"]
    drop_rate = config["model"].get("drop_rate", 0.1)

    model = get_model(
        config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        checkpoint=config["model"].get("checkpoint"),
        drop_rate=drop_rate
    )
    model = model.to(device)

    # Loss function (Section 3.1: Binary Cross-Entropy Loss)
    if num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using Binary Cross-Entropy Loss (BCEWithLogitsLoss)")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using Multi-Class Cross-Entropy Loss")

    # Optimizer (Section 3.1: ADAM optimizer with learning rate 0.001)
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0001)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler_type = config["training"].get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    best_f1 = 0.0

    # Resuming checkpoint
    resume_cfg = config.get("resume", {})
    total_epochs = config["training"]["epochs"]

    if resume_cfg.get("enabled", False) and os.path.exists(resume_cfg.get("checkpoint", "")):
        last_epoch, best_f1 = load_checkpoint(
            resume_cfg["checkpoint"],
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

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6, last_epoch=start_epoch)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0, last_epoch=start_epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Resumed training from epoch {start_epoch + 1}/{total_epochs} (Learning Rate: {current_lr:.6f}, Best F1: {best_f1:.4f})")
    else:
        print(f"Starting training from scratch ({total_epochs} epochs)")

    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()

        running_loss = 0.0

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

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                if num_classes == 1:
                    targets_tensor = labels_device.float().view_as(outputs)
                    loss = criterion(outputs, targets_tensor)
                else:
                    targets_tensor = labels_device.long()
                    loss = criterion(outputs, targets_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        progress_bar.close()
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Validation on holdout validation set
        metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
        print(f"Train Loss : {train_loss:.4f} | Val Loss: {metrics['loss']:.4f}")
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

        # Save History to CSV
        history = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
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