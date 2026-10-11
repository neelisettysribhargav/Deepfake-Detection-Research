import torch
import os

def save_checkpoint(model,optimizer,scheduler,scaler,epoch,metric,best_f1,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metric": metric,
            "best_f1": best_f1
        }, path
    )

def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,device="cpu"):

    print("Loading training checkpoint...")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print("Training checkpoint loaded")

    return (
        checkpoint["epoch"],
        checkpoint["best_f1"]
    )