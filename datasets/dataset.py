import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    """
    Deepfake Face Detection Dataset Loader.
    Labels:
        fake: 0 (Negative class, evaluated via Specificity Sp)
        real: 1 (Positive class, evaluated via Sensitivity Sn)
    Matches the paper's Table 1, Table 2, and Figure 9 definitions.

    Supports:
        1. Manifest loading (Option A, recommended): Instant loading from frozen CSV manifest.
        2. Directory scanning (Legacy fallback): Scans root_dir folder hierarchy.
    """
    def __init__(
        self,
        root_dir=None,
        subsets=None,
        split="train",
        transform=None,
        return_subset=False,
        manifest_path=None
    ):
        self.samples = []
        self.transform = transform
        self.return_subset = return_subset
        self.split = split

        # Auto-detect manifest
        if manifest_path is None and root_dir and str(root_dir).endswith(".csv"):
            manifest_path = root_dir

        if manifest_path is not None and os.path.exists(manifest_path):
            self._load_from_manifest(manifest_path, subsets, split)
        elif root_dir is not None and os.path.exists(root_dir):
            self._load_from_directory(root_dir, subsets, split)
        else:
            raise FileNotFoundError(
                f"Neither valid manifest_path ('{manifest_path}') nor root_dir ('{root_dir}') was found."
            )

    def _load_from_manifest(self, manifest_path, subsets, split):
        subsets_set = set(subsets) if subsets is not None else None
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and row["split"] != split:
                    continue
                if subsets_set is not None and row["subset"] not in subsets_set:
                    continue
                self.samples.append((row["path"], int(row["label"]), row["subset"]))

    def _load_from_directory(self, root_dir, subsets, split):
        classes = {"fake": 0, "real": 1}
        target_subsets = subsets if subsets is not None else ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]
        for subset in target_subsets:
            split_path = os.path.join(root_dir, subset, split)
            for class_name, label in classes.items():
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                for image_name in os.listdir(class_path):
                    if not image_name.startswith(("real_", "fake_")):
                        continue
                    image_path = os.path.join(class_path, image_name)
                    self.samples.append((image_path, label, subset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, subset = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.return_subset:
            return image, label, subset

        return image, label