import sys
import torch
from tqdm import tqdm
from evaluation.metrics import calculate_metrics

def validate(model, loader, criterion, device, return_predictions=False, desc="Validation"):
    """
    Validates model using Binary Cross-Entropy Loss (or Cross-Entropy).
    Includes a stable tqdm progress bar with fixed width (ncols=90) to prevent line wrapping.
    Applies sigmoid threshold (>= 0.5) for binary classification as specified in Section 3.1.
    """
    model.eval()
    running_loss = 0.0

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
        for images, labels in progress_bar:
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

            running_loss += loss.item()
            progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.numpy())

    progress_bar.close()
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0

    metrics = calculate_metrics(targets, predictions)
    metrics['loss'] = avg_loss

    if return_predictions:
        return metrics, targets, predictions
    
    return metrics