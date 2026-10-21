"""
datasets_robust.py

Torch Dataset wrappers around the project's DeepfakeDataset (which must be created with
transform=None so that it returns a PIL image).

  CorruptedTrainDataset  training-time corruption. The corruption plan for sample `index` in
                         epoch `e` depends ONLY on (seed, e, index) via
                         StructuredCorruptor.plan_for -- never on the label.
  ConditionDataset       one fixed evaluation condition (clean, or one corruption at one
                         severity) applied identically for every model, so comparisons between
                         models and arms are paired.
"""

from torch.utils.data import Dataset

from .corruptions import apply_plan, condition_plan


class CorruptedTrainDataset(Dataset):
    def __init__(self, base_ds, post_transform, corruptor=None, pre_transform=None, seed=0):
        self.base = base_ds
        self.post = post_transform
        self.pre = pre_transform          # optional PIL-level transform (used by the generic-augmentation arm)
        self.corruptor = corruptor        # None => no structured corruption
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)           # DataLoader workers are re-created each epoch, so this propagates

    def __len__(self):
        return len(self.base)

    def plan_for(self, index, epoch=None):
        if self.corruptor is None:
            return []
        return self.corruptor.plan_for(self.seed, self.epoch if epoch is None else epoch, index)

    def __getitem__(self, index):
        img, label = self.base[index][:2]
        plan = self.plan_for(index)
        if plan:
            img = apply_plan(img, plan)
        if self.pre is not None:
            img = self.pre(img)
        return self.post(img), label


class ConditionDataset(Dataset):
    def __init__(self, base_ds, post_transform, ctype=None, severity=0, seed=0):
        self.base = base_ds
        self.post = post_transform
        self.ctype = ctype                # None => clean
        self.severity = int(severity)
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, label = self.base[index][:2]
        if self.ctype is not None:
            img = apply_plan(img, condition_plan(self.ctype, self.severity, self.seed, index))
        return self.post(img), label
