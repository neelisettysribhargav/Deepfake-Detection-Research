# Coverage Ablation Plan: Isolating Structure vs. Coverage

**Date**: October 22, 2026  
**Status**: Pre-registered protocol before execution of ablation runs  

---

## 1. Motivation

The 3-seed primary study confirmed that Structured Corruption (Arm C, 96.22% test acc, 0.979 held-out H3 AUC) significantly outperforms Generic Augmentation (Arm B, 93.01% test acc, 0.959 held-out H3 AUC) and Clean Training (Arm A, 90.62% test acc, 0.932 held-out H3 AUC).

However, an important confound exists:
- Where Arm B has overlapping augmentation primitives (color, lighting, occlusion), Arm C's lead is small (+0.005 AUC at severity 3).
- Arm C's largest leads occur on corruption types where Arm B has no direct counterpart (Gaussian noise, JPEG compression, blur past $\sigma=2$).

A reviewer could reasonably ask:
> *"Does Arm C succeed because domain-structured corruption produces superior representations, or simply because Arm B never saw noise and JPEG during training?"*

To isolate **coverage** from **structure**, we pre-register two new control arms across 3 random seeds each (6 runs total, ~10 GPU hours).

---

## 2. Experimental Arms

1. **Arm CnoNJ (`cswin_CnoNJ_s1..3`) — Structure without Extra Coverage**:
   - Arm C corruption pipeline with **Gaussian noise and JPEG excluded** from Family T.
   - Remaining active types: `lighting`, `color_cast`, `gaussian_blur`, `block_occlusion`.
   - Tests whether structured training still beats generic augmentation once noise and JPEG coverage is removed.
2. **Arm BN (`cswin_BN_s1..3`) — Generic Augmentation with Matched Coverage**:
   - Arm B (ColorJitter, GaussianBlur, RandomErasing) **plus Gaussian noise and JPEG**.
   - Applied at the exact marginal frequency ($p \approx 10.8\%$ each) and exact severity tables (1–5) as Arm C.
   - Tests whether generic augmentation catches up to Arm C once coverage is equalized.

---

## 3. Pre-Registered Decision Rules

These rules are fixed before observing any ablation results:

1. **Coverage Explains the Gain**:
   - Criterion: Arm BN is within $0.005$ AUC of Arm C on held-out Family H (severities 3 and 5), AND within $0.5\%$ points of Arm C on official test accuracy.
   - Conclusion: The gain of Arm C was primarily due to exposing the network to noise/JPEG artifacts.
2. **Structure Matters**:
   - Criterion: Arm C beats Arm BN AND Arm CnoNJ beats Arm B, each by at least $\ge 0.010$ AUC on held-out Family H with paired bootstrap 95% confidence intervals strictly excluding zero.
   - Conclusion: Domain-informed structured composition provides representation benefits beyond naive coverage matching.
3. **Partial Attribution (Split Rule)**:
   - If neither extreme holds, report the quantitative attribution split:
     $$\text{Coverage Share} = \frac{\text{AUC}(\text{BN}) - \text{AUC}(\text{B})}{\text{AUC}(\text{C}) - \text{AUC}(\text{B})}$$
     $$\text{Structure Share} = \frac{\text{AUC}(\text{C}) - \text{AUC}(\text{BN})}{\text{AUC}(\text{C}) - \text{AUC}(\text{B})}$$

---

## 4. Methodological Caveats & Disclosures

1. **Sample Distribution Shift in CnoNJ**: Because `p_corrupt = 0.5` is maintained across 4 types instead of 6, each remaining corruption type (blur, lighting, occlusion, color cast) is sampled approximately $1.5\times$ more frequently than in Arm C.
2. **Synthetic Pool Nature**: The evaluation pool is drawn from author test-set images, filtered for baseline noise ($\sigma \le 3.0$). It evaluates corruption invariance, not independent dataset replication.
3. **Frozen Pipeline Guarantee**: The corruption library (`robustness/corruptions.py`) remains byte-identical to `robustness/corruptions_frozen_phase1.py` (verified via SHA-256 before every job).
