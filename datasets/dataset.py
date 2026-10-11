import os
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    """
    Deepfake Face Detection Dataset Loader.
    Labels:
        fake: 0 (Negative class, evaluated via Specificity Sp)
        real: 1 (Positive class, evaluated via Sensitivity Sn)
    Matches the paper's Table 1, Table 2, and Figure 9 definitions.
    """
    def __init__(
        self,
        root_dir,
        subsets,
        split,
        transform=None,
        return_subset=False
    ):
        self.samples = []
        self.transform = transform
        self.return_subset = return_subset

        # 0: Fake (Negative), 1: Real (Positive) as defined in paper Section 3.2 & Fig. 9
        classes = {
            "fake": 0,
            "real": 1
        }

        for subset in subsets:
            split_path = os.path.join(root_dir, subset, split)

            for class_name, label in classes.items():
                class_path = os.path.join(split_path, class_name)

                if not os.path.exists(class_path):
                    continue

                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    self.samples.append((image_path, label, subset))

        self.split = split

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