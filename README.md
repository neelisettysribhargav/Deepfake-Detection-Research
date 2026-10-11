# Deepfake Detection Research using CSWin Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture-CSWin](https://img.shields.io/badge/Architecture-CSWin--Transformer-6f42c1.svg)](https://arxiv.org/abs/2107.00652)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **Deepfake Face Detection** powered by the **Cross-Shaped Window (CSWin) Transformer**. This repository contains the complete research pipeline: training from scratch, pretrained baseline evaluation, Bayesian hyperparameter optimization, and transfer learning on heterogeneous unseen face datasets (DFDC).

---

## 📌 Overview

Traditional Convolutional Neural Networks (CNNs) often suffer from limited receptive fields, making it challenging to capture subtle facial manipulation artifacts across distant regions. Standard Vision Transformers (ViTs) introduce high computational overhead due to full self-attention.

This project implements **CSWin Transformer** (`CSWin_64_24322_small_256`) for deepfake image classification:
- **Cross-Shaped Window Self-Attention**: Computes self-attention in horizontal and vertical stripe windows, enabling global context modeling at low computational complexity.
- **Locally-enhanced Positional Encoding (LePE)**: Captures fine-grained spatial inductive biases essential for identifying high-frequency deepfake blending boundaries.
- **Multi-Dataset Benchmarking**: Evaluates sensitivity ($S_n$), specificity ($S_p$), precision, recall, and F1-score across distinct facial manipulation subsets.
- **Bayesian Optimization**: Uses Optuna to discover optimal learning rates and dropout configurations.
- **Heterogeneous Transfer Learning**: Evaluates cross-dataset domain adaptation using a frozen CSWin backbone fine-tuned on mixed subsets and DFDC.

---

## 📂 Project Structure

```text
Deepfake-Detection-Research/
├── configs/
│   └── cswin_baseline.yaml        # Central configuration (data, model, training)
├── datasets/
│   ├── __init__.py
│   ├── dataset.py                 # Multi-subset DeepfakeDataset loader
│   └── transforms.py              # 256x256 resizing, normalization, augmentations
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Accuracy, Precision, Recall, F1, Confusion Matrix
│   └── test.py                    # Multi-dataset benchmark evaluation
├── experiments/
│   ├── logs/                      # Training history & loss curves (CSV)
│   └── results/                   # Classification reports and confusion matrices
├── models/
│   ├── __init__.py
│   ├── model_factory.py           # Model instantiator
│   └── cswin/                     # CSWin Transformer architecture
│       ├── attention.py           # Cross-shaped window self-attention
│       ├── block.py               # CSWin blocks
│       ├── cswin.py               # Complete CSWin Transformer backbone
│       ├── lepe.py                # Locally-enhanced Positional Encoding
│       ├── patch_embed.py         # Convolutional patch embedding
│       └── patch_merge.py         # Patch merging layers
├── training/
│   ├── __init__.py
│   ├── train.py                   # Full training pipeline (CosineAnnealingLR, BCE loss)
│   ├── validate.py                # Validation and epoch metrics computation
│   ├── tune.py                    # Optuna Bayesian hyperparameter optimization
│   └── finetune.py                # Transfer learning / head fine-tuning on DFDC
├── utils/
│   ├── checkpoint.py              # Model checkpoint save/load utilities
│   ├── logger.py                  # Training loggers
│   └── seed.py                    # Deterministic seed configuration
├── main.py                        # Unified command-line interface entry point
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## 📊 Benchmark Results

### 1. CSWin Trained from Scratch (`cswin_scratch`)
Evaluated across individual benchmark subsets at 256×256 resolution:

| Benchmark Subset | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset 1** | **87.35%** | 85.93% | 89.44% | **87.65%** |
| **Dataset 2** | **86.97%** | 86.29% | 88.03% | **87.15%** |
| **Dataset 3** | **87.14%** | 86.85% | 87.65% | **87.25%** |
| **Dataset 4** | **89.48%** | 89.40% | 89.67% | **89.53%** |

### 2. Pretrained CSWin Baseline (`cswin_pretrained`)

| Benchmark Subset | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset 1** | 85.56% | 82.16% | 90.96% | 86.34% |
| **Dataset 2** | 85.40% | 86.47% | 84.06% | 85.25% |
| **Dataset 3** | 86.53% | 86.08% | 87.27% | 86.67% |
| **Dataset 4** | **89.92%** | 88.14% | 92.34% | **90.19%** |

*All results include per-class confusion matrices and scikit-learn classification reports saved under `experiments/results/`.*

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/neelisettysribhargav/Deepfake-Detection-Research.git
cd Deepfake-Detection-Research
```

Using Conda:
```bash
conda create -n cswin_baseline python=3.10 -y
conda activate cswin_baseline
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Organize your dataset directory structure as follows:

```text
data/
└── Dataset-1/
    ├── Data Set 1/
    │   ├── train/
    │   │   ├── real/
    │   │   └── fake/
    │   ├── validation/
    │   │   ├── real/
    │   │   └── fake/
    │   └── test/
    │       ├── real/
    │       └── fake/
    ├── Data Set 2/ ...
    ├── Data Set 3/ ...
    └── Data Set 4/ ...
```

> **Note**: Labels follow standard convention: `0` = Fake (Negative class), `1` = Real (Positive class).

---

## 🛠 Usage & CLI Execution

All experiments are controlled via `main.py` using the `--mode` argument:

### 1. Train Baseline Model
Train the CSWin Transformer model from scratch using the settings defined in `configs/cswin_baseline.yaml`:
```bash
python main.py --mode train
```

### 2. Evaluate / Test
Evaluate the best checkpoint across all test datasets:
```bash
python main.py --mode test
```
*Outputs classification reports and confusion matrices directly to `experiments/results/<experiment_name>/`.*

### 3. Bayesian Hyperparameter Tuning
Run Bayesian optimization with Optuna over learning rate and dropout rate:
```bash
python main.py --mode tune
```

### 4. Transfer Learning / Fine-Tuning
Freeze the CSWin backbone and fine-tune the classification head on unseen heterogeneous data (e.g. mixed Dataset-1 + DFDC):
```bash
python main.py --mode finetune
```

---

## ⚙️ Configuration

Experiments are configured in [`configs/cswin_baseline.yaml`](configs/cswin_baseline.yaml):

```yaml
experiment:
  name: cswin_scratch_v2

model:
  name: CSWin_64_24322_small_256
  num_classes: 1
  pretrained: false
  drop_rate: 0.1

data:
  root: data/Dataset-1
  image_size: 256
  train_subsets: [Data Set 1, Data Set 2, Data Set 3, Data Set 4]

training:
  epochs: 30
  batch_size: 32
  optimizer: Adam
  learning_rate: 0.001
  loss: BCEWithLogitsLoss
  scheduler: cosine
```

---

## 🔬 Citation & References

- **CSWin Transformer**: Dong et al., *"CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows"*, CVPR 2022. [[arXiv:2107.00652](https://arxiv.org/abs/2107.00652)]
- **Deepfake Detection Challenge (DFDC)**: Dolhansky et al., *"The Deepfake Detection Challenge Dataset"*, [[arXiv:2006.07397](https://arxiv.org/abs/2006.07397)]

---

## 📄 License

This research codebase is distributed under the MIT License.
