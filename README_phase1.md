# Phase 1 Toolkit: Multi-Axis Robustness Study

This toolkit implements the label-free corruption library, paired evaluation conditions, dataset wrappers, shortcut verification audits, and training scripts for the multi-axis robustness study (Arms A, B, and C).

---

## 1. Directory Structure

```text
Deepfake-Detection-Research/
├── robustness/
│   ├── __init__.py
│   ├── corruptions.py          # 11 deterministic, label-free corruptions across 5 severities
│   └── datasets_robust.py      # CorruptedTrainDataset & ConditionDataset wrappers
├── scripts/
│   ├── preview_corruptions.py  # Visual QA grid (1 row/corruption, columns/severity)
│   ├── make_dfdc_manifest.py   # Cross-dataset DFDC face frames manifest generator
│   ├── make_robust_pool.py     # Frozen 2,000-image low-noise test pool builder
│   ├── shortcut_check.py       # Independence audit & logistic metadata classifier
│   ├── evaluate_robustness.py  # Paired multi-condition evaluation (acc, bal_acc, Sn, Sp, AUC, ECE, Brier)
│   └── train_arms.py           # Unified training script for Arms A, B, and C
└── README_phase1.md            # This operational run guide
```

---

## 2. Corruption Families

* **Family T (Training Family)**: Used for training Arm C and for synthetic validation.
  * `lighting`, `color_cast`, `gaussian_blur`, `block_occlusion`, `gaussian_noise`, `jpeg`
* **Family H (Held-out Family)**: **NEVER** used in training. Reserved strictly for evaluating out-of-distribution generalisation.
  * `salt_pepper`, `pixelate`, `motion_blur`, `color_distort`, `overlay_occluder`

### Core Design Principles
1. **Label-Free**: Corruption sampling never observes the ground-truth label. The plan is generated strictly from `(seed, epoch, sample_index)`.
2. **Paired Evaluation**: Evaluation conditions apply identical plans to the frozen 2,000-image pool across all models.
3. **Monotonic Severity**: Severity levels $1 \dots 5$ strictly increase perturbation magnitude.

---

## 3. Operational Workflow (Steps 1 – 5)

### Step 1: Self-Test Corruption Library
```powershell
python -m robustness.corruptions
# Expected Output: SELF-TEST PASSED
```

### Step 2: Visual QA Preview Grid
Generate visual verification sheets across all 11 corruption types at severities 1, 3, and 5:
```powershell
python scripts/preview_corruptions.py --image "data/Dataset-1/Data Set 1/train/real/real_54999.jpg" "data/Dataset-1/Data Set 1/train/fake/fake_54999.jpg" --out experiments/robustness/preview.png
# Generates experiments/robustness/preview.png (1430x1760)
```

### Step 3: Manifest & Pool Generation
1. **DFDC Cross-Dataset Manifest**:
   ```powershell
   python scripts/make_dfdc_manifest.py --root data/Dataset-2 --out data/manifests/dfdc_faces.csv
   # Result: 1,530 frames (770 fake, 760 real across 153 videos)
   ```
2. **Frozen Robust Evaluation Pool**:
   Filter test split for images with natural noise $\sigma \le 3.0$ and sample 1,000 real and 1,000 fake images:
   ```powershell
   python scripts/make_robust_pool.py --manifest data/manifests/author_data.csv --features experiments/inspect/image_features.csv --out data/manifests/robust_pool.csv
   # Result: 2,000 images balanced across subsets
   ```

### Step 4: Shortcut Audit
Verify that the corruption plan contains zero mutual information with image labels:
```powershell
# Standard audit (must print PASS):
python scripts/shortcut_check.py --manifest data/manifests/author_data.csv --epochs 3
# Expected: VERDICT: PASS

# Negative control with deliberate 10% bias (must print FAIL):
python scripts/shortcut_check.py --manifest data/manifests/author_data.csv --negative-control
# Expected: VERDICT: FAIL
```

### Step 5: Robustness Profiling & Arm Training
1. **Baseline Profiling (Before Picture)**:
   ```powershell
   python scripts/evaluate_robustness.py --checkpoint experiments/finetune/cswin_pre_a/best.pth --tag cswin_pre_a --model CSWin_64_24322_small_256 --pool data/manifests/robust_pool.csv --dfdc data/manifests/dfdc_faces.csv
   ```
2. **Arm Pilot / Study**:
   * **Arm A (Clean Control)**:
     ```powershell
     python scripts/train_arms.py --arm A --seed 1 --weights checkpoints/cswin/cswin_small_224.pth
     ```
   * **Arm B (Generic Augmentation)**:
     ```powershell
     python scripts/train_arms.py --arm B --seed 1 --weights checkpoints/cswin/cswin_small_224.pth
     ```
   * **Arm C (Structured Corruption)**:
     ```powershell
     python scripts/train_arms.py --arm C --seed 1 --weights checkpoints/cswin/cswin_small_224.pth
     ```

---

## 4. Three Go/No-Go Gates (Arm C Pilot)

Before scaling to the full 27 GPU-hour study, agree with your advisor on these pre-specified gates:

1. **Gate 1 (Clean Retention)**: Clean validation / test accuracy for Arm C must not degrade by more than 1.5% compared to Arm A.
2. **Gate 2 (Corruption Invariance)**: Balanced accuracy and AUC on held-out Family H corruptions must improve significantly ($\ge 5$ percentage points) over Arm A.
3. **Gate 3 (Calibration & Sensitivity Balance)**: Real Sensitivity ($S_n$) on degraded images must show substantial recovery without collapsing Fake Specificity ($S_p$), with Brier scores and ECE remaining bounded.
