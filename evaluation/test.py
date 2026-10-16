import sys
import os
import csv
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DeepfakeDataset, get_test_transform
from models.model_factory import get_model
from training.validate import validate

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')) else "configs/cswin_baseline.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    exp_name = config["experiment"]["name"]
    dest_dir = f"experiments/results/{exp_name}"
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    num_classes = config["model"]["num_classes"]
    drop_rate = config["model"].get("drop_rate", 0.1)
    drop_path_rate = config["model"].get("drop_path_rate", 0.1)

    model = get_model(
        config["model"]["name"],
        num_classes=num_classes,
        pretrained=False,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )

    checkpoint_dir = f"{config['paths']['checkpoint_dir']}/{exp_name}"
    candidate_paths = [
        config["model"].get("checkpoint"),
        f"{checkpoint_dir}/best_e10_frozen.pth",
        f"{checkpoint_dir}/best.pth"
    ]
    best_checkpoint_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if best_checkpoint_path:
        print(f"Loading best checkpoint from: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print("Warning: No checkpoint found. Running with initialized weights.")

    model = model.to(device)
    model.eval()

    if num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    results = []
    os.makedirs(f"{dest_dir}/confusion_matrix", exist_ok=True)
    os.makedirs(f"{dest_dir}/classification_reports", exist_ok=True)

    manifest_path = config["data"].get("manifest", None)

    for dataset_name, subsets in config["data"]["test_subsets"].items():
        print(f"\n==========================================")
        print(f"Testing on {dataset_name} ({subsets})")
        print(f"==========================================")

        test_dataset = DeepfakeDataset(
            root_dir=config["data"].get("root"),
            subsets=subsets,
            split="test",
            transform=get_test_transform(config["data"]["image_size"]),
            manifest_path=manifest_path
        )

        if len(test_dataset) == 0:
            print(f"Skipping {dataset_name}: 0 benchmark test samples found (Table 1 test set is distributed across Data Sets 1, 3, and 4).")
            continue

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False
        )

        metrics, targets, predictions = validate(
            model, test_loader, criterion, device, return_predictions=True, desc=f"Testing {dataset_name}"
        )

        cm = confusion_matrix(targets, predictions, labels=[0, 1])
        target_names = ["Fake (0)", "Real (1)"]
        cf_report = classification_report(targets, predictions, target_names=target_names, zero_division=0)

        print(f"Accuracy         : {metrics['accuracy'] * 100:.2f}% (Paper Target: 98.79%)")
        print(f"Sensitivity (Sn) : {metrics['sensitivity'] * 100:.2f}% (Paper Target: 99.23%)")
        print(f"Specificity (Sp) : {metrics['specificity'] * 100:.2f}% (Paper Target: 98.28%)")
        print(f"F1-Score         : {metrics['f1'] * 100:.2f}% (Paper Target: 98.72%)")
        print(f"Precision        : {metrics['precision'] * 100:.2f}%")
        print("\nConfusion Matrix [Row: True (Fake, Real), Col: Pred (Fake, Real)]:")
        print(cm)
        print("\nClassification Report:")
        print(cf_report)

        with open(f"{dest_dir}/confusion_matrix/{dataset_name}.txt", "w") as file:
            file.write(str(cm))

        with open(f"{dest_dir}/classification_reports/{dataset_name}.txt", "w") as file:
            file.write(cf_report)

        results.append({
            "dataset": dataset_name,
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    with open(f"{dest_dir}/test_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTesting finished. All results saved to {dest_dir}")

if __name__ == "__main__":
    main()