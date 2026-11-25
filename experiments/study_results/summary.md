# Study summary

Runs: {'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [1, 2, 3]}, bootstrap resamples: 500

## Official test (all subsets), mean +/- std over seeds

| Arm | Acc % | AUC | Sn % | Sp % |
|---|---|---|---|---|
| A | 90.62 +/- 0.18 | 0.9861 +/- 0.0002 | 83.4 +/- 0.6 | 97.8 +/- 0.2 |
| B | 93.01 +/- 0.21 | 0.9902 +/- 0.0011 | 88.1 +/- 0.5 | 97.9 +/- 0.2 |
| C | 96.22 +/- 0.08 | 0.9950 +/- 0.0002 | 94.1 +/- 0.1 | 98.3 +/- 0.0 |

## Synthetic pool: mean AUC over corruption types, mean +/- std over seeds

| Family | Severity | A | B | C |
|---|---|---|---|---|
| T | 1 | 0.9892 +/- 0.0004 | 0.9925 +/- 0.0012 | 0.9961 +/- 0.0003 |
| T | 3 | 0.9677 +/- 0.0015 | 0.9833 +/- 0.0018 | 0.9914 +/- 0.0002 |
| T | 5 | 0.9001 +/- 0.0072 | 0.9338 +/- 0.0023 | 0.9757 +/- 0.0010 |
| H | 1 | 0.9795 +/- 0.0007 | 0.9885 +/- 0.0013 | 0.9944 +/- 0.0004 |
| H | 3 | 0.9316 +/- 0.0032 | 0.9593 +/- 0.0027 | 0.9793 +/- 0.0013 |
| H | 5 | 0.8274 +/- 0.0070 | 0.8649 +/- 0.0077 | 0.9169 +/- 0.0025 |

## Paired bootstrap differences (95% CI)

| Dataset | Metric | Comparison | Diff | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| official | auc | B-A | +0.0042 | [+0.0035, +0.0048] | yes |
| official | auc | C-A | +0.0090 | [+0.0080, +0.0100] | yes |
| official | auc | C-B | +0.0048 | [+0.0041, +0.0055] | yes |
| official | acc | B-A | +0.0239 | [+0.0213, +0.0264] | yes |
| official | acc | C-A | +0.0560 | [+0.0521, +0.0601] | yes |
| official | acc | C-B | +0.0321 | [+0.0289, +0.0353] | yes |
| official | sn | B-A | +0.0470 | [+0.0421, +0.0517] | yes |
| official | sn | C-A | +0.1076 | [+0.0994, +0.1155] | yes |
| official | sn | C-B | +0.0606 | [+0.0550, +0.0668] | yes |
| pool_clean | auc | B-A | +0.0022 | [+0.0013, +0.0033] | yes |
| pool_clean | auc | C-A | +0.0052 | [+0.0037, +0.0072] | yes |
| pool_clean | auc | C-B | +0.0030 | [+0.0018, +0.0044] | yes |
| pool_T3 | auc | B-A | +0.0156 | [+0.0136, +0.0176] | yes |
| pool_T3 | auc | C-A | +0.0237 | [+0.0206, +0.0270] | yes |
| pool_T3 | auc | C-B | +0.0081 | [+0.0065, +0.0098] | yes |
| pool_H3 | auc | B-A | +0.0277 | [+0.0247, +0.0308] | yes |
| pool_H3 | auc | C-A | +0.0477 | [+0.0432, +0.0522] | yes |
| pool_H3 | auc | C-B | +0.0200 | [+0.0175, +0.0230] | yes |
| pool_H5 | auc | B-A | +0.0375 | [+0.0336, +0.0413] | yes |
| pool_H5 | auc | C-A | +0.0896 | [+0.0838, +0.0954] | yes |
| pool_H5 | auc | C-B | +0.0521 | [+0.0473, +0.0569] | yes |
| dfdc | auc | B-A | +0.0582 | [+0.0286, +0.0871] | yes |
| dfdc | auc | C-A | +0.0182 | [-0.0155, +0.0501] | no |
| dfdc | auc | C-B | -0.0400 | [-0.0720, -0.0079] | yes |

## Pre-specified gates (arm C)

1. C beats B on held-out family H, mean AUC at severity 3: diff +0.0200 [+0.0175, +0.0230] -> PASS
2. Clean dedup-validation accuracy of C within 1 point of A: A 98.61%, C 98.69% -> PASS
3. DFDC is reported as a secondary result (baseline AUC is near chance).
