# Deepfake Detection Research using CSWin Transformer & Robustness Benchmarking

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture-CSWin](https://img.shields.io/badge/Architecture-CSWin--Transformer-6f42c1.svg)](https://arxiv.org/abs/2107.00652)
[![License-MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **Deepfake Face Detection** using the **Cross-Shaped Window (CSWin) Transformer**, incorporating multi-backbone benchmarking, empirical replication audits, forensic distribution-shift analysis, and corruption-robustness training.

---

## 📌 Overview & Research Findings

This repository investigates the reproducibility and real-world robustness of Vision Transformers for deepfake detection, focusing on the CSWin Transformer (`CSWin_64_24322_small_256`, 35M parameters) evaluated against top CNN and hybrid architectures (ConvNeXt-Tiny, Xception, InceptionV3).

### Key Empirical Findings
1. **Validation Replication Confirmed (98.52% vs. 99.1% Claimed)**:  
   Fine-tuning CSWin-S initialized with ImageNet-1K weights achieves **98.52% accuracy** (AUC **0.9985**) on the deduplicated validation benchmark, confirming the high validation accuracy claimed in the literature.
2. **The Test Performance Deficit (~90% Hard Ceiling)**:  
   On the official holdout test set (10,905 images), performance drops across all models to **89.5% – 90.7%**, driven entirely by an asymmetric drop in **Real Sensitivity** (~80% test vs. ~98.5% val), while **Fake Specificity** remains at ~98.5%.
3. **Forensic Discovery: Asymmetric Test Corruption Shift ($p < 10^{-46}$)**:  
   - **Sharpness & Compression**: The 502 test reals misclassified by all models are **4.6× to 7.1× sharper** than training reals (median Laplacian variance 887.67 vs. 124.79, $p = 8.19 \times 10^{-47}$) and less compressed (14.07 KB vs. 8.95 KB, $p = 5.77 \times 10^{-49}$).
   - **Noise Shift**: Exactly **0.0%** of train and validation real images have Immerkaer noise $\sigma > 6$, whereas **3.8% of test reals overall (and 4.5% in Data Set 1)** exhibit heavy speckle noise ($\sigma > 6$).
   - Models trained on clean data learn that high-frequency noise and sharp edge detail represent synthetic artifacts, causing them to falsely classify noisy, sharp real faces as deepfakes.

For the full statistical and diagnostic report, see [**`docs/ANALYSIS_REPORT.md`**](file:///c:/Users/B/Docs/Projects/Deepfake-Detection-Research/docs/ANALYSIS_REPORT.md).

---

## 📊 Benchmark Results

### 1. Model Comparison Matrix (Official Table 1 Benchmark)

| Model Architecture | Parameters | Dedup-Val Acc | Dedup-Val AUC | Test Acc | Test $S_n$ (Real) | Test $S_p$ (Fake) | Test AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSWin-S (Pretrained)** | 35M | **98.52%** | **0.9985** | **89.54%** | 80.49% | **98.45%** | **0.9830** |
| **ConvNeXt-Tiny (Pretrained)**| 28M | **98.26%** | **0.9985** | **90.40%** | **82.73%** | 97.96% | **0.9845** |
| **Xception (Pretrained)** | 23M | 97.92% | 0.9976 | 88.94% | 80.95% | 96.81% | 0.9695 |
| **InceptionV3 (Pretrained)** | 27M | 98.24% | 0.9980 | 69.35% | 40.44% | 97.85% | 0.8345 |
| **CSWin from Scratch (e10)** | 35M | 94.38% | 0.9856 | **90.68%** | **87.42%** | 93.90% | 0.9691 |

### 2. Multi-Model Ensembles
- **Dual Ensemble (CSWin-S + ConvNeXt-Tiny)**:
  - **Dedup-Val Accuracy**: **98.68%** (AUC **0.9990**)
  - **Holdout Test Accuracy**: **90.58%** (AUC **0.9868** — highest AUC achieved across all benchmarks)
- **Tri-Ensemble (CSWin-S + ConvNeXt-Tiny + Xception)**:
  - **Dedup-Val Accuracy**: **98.65%** (AUC **0.9989**)
  - **Holdout Test Accuracy**: **90.22%** (AUC **0.9855**)

---

## 📂 Repository Structure

```text
Deepfake-Detection-Research/
├── configs/
│   └── cswin_baseline.yaml         # Central baseline configuration
├── data/
│   ├── manifests/                  # Benchmark splits & deduplication manifests
│   │   ├── author_data.csv         # Full 140k dataset manifest
│   │   ├── author_data_val_dedup.csv # 37.5k deduplicated validation benchmark
│   │   └── author_data_test.csv    # 10.9k official holdout test manifest
│   ├── Dataset-1/                  # Primary 4-subset facial manipulation images
│   └── Dataset-2/                  # Unseen DFDC video frames for cross-dataset testing
├── datasets/
│   ├── dataset.py                  # DeepfakeDataset PyTorch dataset class
│   └── transforms.py               # Image transforms and normalization
├── docs/
│   └── ANALYSIS_REPORT.md          # Comprehensive audit, pilot, and diagnostic report
├── evaluation/
│   ├── metrics.py                  # Metric computation (Accuracy, Sn, Sp, F1, Precision)
│   └── test.py                     # Multi-dataset benchmark evaluation
├── experiments/
│   ├── audit_results/              # Perceptual hash leakage & threshold sweep logs
│   ├── checkpoints/                # Canonical frozen scratch checkpoints (best_e10_frozen.pth)
│   ├── finetune/                   # Best checkpoints for fine-tuned models
│   ├── inspect/                    # Forensic inspection stats, features & contact sheets
│   ├── pilot_results/              # 8-factor pilot ablation summaries and training histories
│   └── predictions/                # Per-image predictions and evaluation metric CSVs
├── models/
│   ├── model_factory.py            # Unified backbone factory (CSWin, ConvNeXt, Xception)
│   └── cswin/                      # Native CSWin Transformer architecture
├── scripts/
│   ├── evaluate.py                 # Unified evaluation script for predictions & metrics
│   ├── finetune.py                 # Pretrained backbone fine-tuning script
│   ├── inspect_misses.py           # Consensus misclassification forensic analyzer
│   ├── load_cswin.py               # Official ImageNet weight remap and loader
│   └── measure_noise.py            # Noise & corruption prevalence estimator across 56k images
├── training/
│   ├── train.py                    # Training engine
│   ├── validate.py                 # Evaluation engine
│   ├── tune.py                     # Bayesian hyperparameter optimization
│   └── finetune.py                 # Transfer learning / head fine-tuning
├── utils/
│   ├── checkpoint.py               # Checkpoint I/O
│   ├── logger.py                   # CSV training logger
│   └── seed.py                     # Deterministic random seed utilities
├── main.py                         # CLI entry point
├── requirements.txt                # Package dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/neelisettysribhargav/Deepfake-Detection-Research.git
cd Deepfake-Detection-Research

# Activate the conda environment
conda activate cswin_baseline

# Or install dependencies via pip
pip install -r requirements.txt
```

### 2. Reproducing Evaluations on Any Checkpoint

To evaluate any model checkpoint on the deduplicated validation and test sets:

```bash
# Evaluate Pretrained CSWin-S
python scripts/evaluate.py \
    --checkpoint experiments/finetune/cswin_pre_a/best.pth \
    --tag cswin_pre_a \
    --model cswin_small_224 \
    --dedup-manifest data/manifests/author_data_val_dedup.csv

# Evaluate Pretrained ConvNeXt-Tiny
python scripts/evaluate.py \
    --checkpoint experiments/finetune/convnext_tiny_pre/best.pth \
    --tag convnext_tiny_pre \
    --model convnext_tiny \
    --dedup-manifest data/manifests/author_data_val_dedup.csv

# Evaluate Canonical Epoch-10 Frozen Scratch Baseline
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/cswin_scratch_v3/best_e10_frozen.pth \
    --tag e10 \
    --dedup-manifest data/manifests/author_data_val_dedup.csv
```

### 3. Running Diagnostic Analysis Scripts

```bash
# Analyze consensus-missed test reals vs. correct images (sharpness, size, brightness)
python scripts/inspect_misses.py \
    --preds experiments/predictions/cswin_pre_a_predictions.csv experiments/predictions/convnext_tiny_pre_predictions.csv \
    --manifest data/manifests/author_data.csv \
    --out experiments/inspect

# Measure noise (Immerkaer sigma) and corruption prevalence across all 56k images
python scripts/measure_noise.py \
    --manifest data/manifests/author_data.csv \
    --out experiments/inspect
```

---

## 🔬 Next Phase: Corruption Robustness Training

Having isolated the failure mode to high-frequency speckle noise and uncompressed edge details in holdout real faces, the next phase focuses on **Corruption-Robust Training**:

- **Arm A (Clean Baseline)**: Frozen at 98.5% val / 89.5%–90.7% test.
- **Arm B (Generic Augmentation)**: Random horizontal flip, cropping, and color jitter.
- **Arm C (Corruption-Robust Training)**: 50% stochastic corruptions sampled equally for real and fake faces:
  - Gaussian noise injection ($\sigma \in [2, 12]$)
  - JPEG compression artifact jitter ($Q \in [40, 95]$)
  - Gaussian blur and unsharp sharpening filter jitter
  - Random rectangular synthetic occluders
- **Generalization Gate**: Evaluation on unseen DFDC faces (`data/Dataset-2`) and synthetic corruption severity sweeps.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).
Official CSWin Transformer architecture adapted from [Microsoft CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer).
