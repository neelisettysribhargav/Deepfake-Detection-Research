import os
import sys
import yaml
import torch
import numpy as np
import optuna
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DeepfakeDataset, get_train_transform, get_test_transform
from models.model_factory import get_model
from training.validate import validate
from utils.seed import set_seed

"""
Bayesian Optimization Hyperparameter Tuning for CSWin Transformer.
Matches Section 4 of the paper:
"The Bayesian optimization-based hyper-parameter tuning technique is incorporated
in this research work while training the underlying deep learning models [28]."

Tunes:
- learning_rate: between 1e-4 and 5e-3 (paper baseline: 0.001)
- drop_rate: between 0.1 and 0.5 (paper Section 3.1: dropout rate of 0.5 to 0.1)
"""

def objective(trial, config, train_subset, val_subset, device):
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    drop_rate = trial.suggest_float("drop_rate", 0.1, 0.5, step=0.05)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = get_model(
        config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        drop_rate=drop_rate
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    eval_epochs = 5
    for epoch in range(eval_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    metrics = validate(model, val_loader, criterion, device)
    return metrics["f1"]

def main():
    set_seed(42)
    with open("configs/cswin_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")

    # Load 10% subset for fast tuning
    full_train = DeepfakeDataset(
        root_dir=config["data"]["root"],
        subsets=config["data"]["train_subsets"],
        split="train",
        transform=get_train_transform(config["data"]["image_size"])
    )
    indices = np.random.choice(len(full_train), size=min(5000, len(full_train)), replace=False)
    train_subset = Subset(full_train, indices)

    val_dataset = DeepfakeDataset(
        root_dir=config["data"]["root"],
        subsets=config["data"]["val_subsets"],
        split="validation",
        transform=get_test_transform(config["data"]["image_size"])
    )
    val_indices = np.random.choice(len(val_dataset), size=min(1000, len(val_dataset)), replace=False)
    val_subset = Subset(val_dataset, val_indices)

    print("Running Bayesian Optimization using Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, train_subset, val_subset, device), n_trials=10)
    print("Best trial:")
    print(f"  F1-Score: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
