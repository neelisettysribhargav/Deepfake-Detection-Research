import sys
import torch
from tqdm import tqdm
from evaluation.metrics import calculate_metrics

def validate(model, loader, criterion, device, return_predictions=False, desc="Validation"):
    """
    Validates model using Binary Cross-Entropy Loss (or Cross-Entropy).
    Includes a stable tqdm progress bar with fixed width (ncols=100) to prevent line wrapping.
    Applies sigmoid threshold (>= 0.5) for binary classification as specified in Section 3.1.
    Optimized to eliminate per-batch CPU-GPU sync stalls by batching tensor transfers.
    """
    model.eval()
    running_loss = torch.tensor(0.0, device=device)

    predictions = []
    targets = []

    progress_bar = tqdm(
        loader,
        desc=desc,
        total=len(loader),
        ncols=100,
        bar_format="{desc:<12} {percentage:3.0f}%|{bar:16}| {n_fmt:>4}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        mininterval=0.5,
        leave=False,
        file=sys.stdout
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)

                # Check if binary (single logit) or multi-class
                if outputs.shape[-1] == 1 or outputs.ndim == 1 or isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    targets_tensor = labels_device.float().view_as(outputs)
                    loss = criterion(outputs, targets_tensor)
                    preds = (torch.sigmoid(outputs) >= 0.5).long().view(-1)
                else:
                    targets_tensor = labels_device.long()
                    loss = criterion(outputs, targets_tensor)
                    preds = torch.argmax(outputs, dim=1)

            running_loss += loss.detach()
            predictions.append(preds.detach())
            targets.append(labels)

            if batch_idx % 25 == 0 or (batch_idx + 1) == len(loader):
                progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

    progress_bar.close()
    avg_loss = (running_loss / len(loader)).item() if len(loader) > 0 else 0.0

    # Single bulk PCIe transfer at the very end
    all_predictions = torch.cat(predictions).cpu().numpy()
    all_targets = torch.cat(targets).numpy()

    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = avg_loss

    if return_predictions:
        return metrics, all_targets, all_predictions
    
    return metrics