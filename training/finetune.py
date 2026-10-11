import os
import sys
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DeepfakeDataset, get_train_transform, get_test_transform
from models.model_factory import get_model
from training.validate import validate
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

"""
Transfer Learning / Fine-Tuning Module for Unseen Heterogeneous Data.
Matches Section 2.1 and Section 4 (Table 7) of the paper:
1. Loads trained CSWin Transformer weights from baseline.
2. Freezes backbone weights (Trainable = False / requires_grad = False).
3. Mixed training dataset:
   - 20% subset of Deepfake Face Detection training dataset
   - Deepfake Detection Challenge Dataset - Face Images (Dataset-2)
4. Fine-tunes only the top classification head for 10 epochs.
5. Evaluates on Test Dataset-1 and Test Dataset-6 (DFDC).
"""

def main():
    set_seed(42)
    with open("configs/cswin_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    image_size = config["data"]["image_size"]

    print("=== Paper Transfer Learning / Fine-Tuning Setup ===")

    # 1. Load 20% subset of Deepfake Face Detection training dataset (Dataset-1)
    full_ds1 = DeepfakeDataset(
        root_dir=config["data"]["root"],
        subsets=config["data"]["train_subsets"],
        split="train",
        transform=get_train_transform(image_size)
    )
    n_20pct = int(0.20 * len(full_ds1))
    indices_20 = np.random.choice(len(full_ds1), size=n_20pct, replace=False)
    subset_ds1 = Subset(full_ds1, indices_20)
    print(f"20% subset of Dataset-1: {len(subset_ds1)} samples")

    # 2. Check if Dataset-2 (Deepfake Detection Challenge Dataset) exists
    ds2_dir = "data/Dataset-2"
    if os.path.exists(ds2_dir):
        # Load Dataset-2 images
        ds2_samples = []
        for class_name, label in [("fake", 0), ("real", 1)]:
            class_folder = os.path.join(ds2_dir, class_name)
            if os.path.exists(class_folder):
                for img_name in os.listdir(class_folder):
                    ds2_samples.append((os.path.join(class_folder, img_name), label, "DFDC"))
        print(f"Dataset-2 (DFDC) images loaded: {len(ds2_samples)}")

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                from PIL import Image
                p, l, _ = self.samples[idx]
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, l

        ds2_train = SimpleDataset(ds2_samples, get_train_transform(image_size))
        mixed_train_dataset = ConcatDataset([subset_ds1, ds2_train])
    else:
        print("Warning: data/Dataset-2 not found. Fine-tuning with 20% Dataset-1 subset only.")
        mixed_train_dataset = subset_ds1

    print(f"Total mixed fine-tuning training instances: {len(mixed_train_dataset)}")

    train_loader = DataLoader(
        mixed_train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Instantiate model and load baseline weights
    model = get_model(
        config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        drop_rate=config["model"].get("drop_rate", 0.1)
    )

    checkpoint_path = f"{config['paths']['checkpoint_dir']}/{config['experiment']['name']}/best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained baseline weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Warning: Trained baseline checkpoint not found. Starting from scratch weights.")

    # 4. Freeze core model weights (Trainable = False as stated in Section 4)
    print("Freezing core model backbone layers (Trainable = False)...")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the top classification head
    for param in model.head.parameters():
        param.requires_grad = True

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Trainable parameters are only in the head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    # 5. Fine-tune for exactly 10 epochs (Section 4: "fine-tuned for 10 epochs")
    finetune_epochs = 10
    print(f"Beginning fine-tuning for {finetune_epochs} epochs...")

    for epoch in range(finetune_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{finetune_epochs}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{finetune_epochs} completed. Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    ft_save_dir = f"{config['paths']['checkpoint_dir']}/cswin_finetuned"
    os.makedirs(ft_save_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, f"{ft_save_dir}/finetuned_best.pth")
    print(f"Fine-tuned model checkpoint saved to {ft_save_dir}/finetuned_best.pth")

if __name__ == "__main__":
    main()
