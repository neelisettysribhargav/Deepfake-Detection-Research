"""
make_figures.py

Draws the study figures from the tables written by aggregate_study.py:

  fig1_auc_vs_severity.png   AUC vs severity for every corruption type, one line per arm (mean +/- std over seeds)
  fig2_family_severity.png   mean AUC over types per family and severity, per arm
  fig3_sn_sp_by_type.png     sensitivity (real recall) and specificity (fake recall) at one severity, per type
  fig4_official_test.png     official test accuracy, Sn and Sp per arm (dots = seeds)
  fig5_calibration.png       expected calibration error per arm (clean pool, T@3, H@3, H@5)

  python scripts/make_figures.py --dir experiments/study_results
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#d62728", "BN": "#17becf", "CnoNJ": "#ff7f0e"}
NAMES = {"A": "A: clean", "B": "B: generic aug.", "C": "C: structured corruption",
         "BN": "B + noise/JPEG", "CnoNJ": "C minus noise/JPEG"}
TYPE_ORDER = ["lighting", "color_cast", "gaussian_blur", "block_occlusion", "gaussian_noise", "jpeg",
              "salt_pepper", "pixelate", "motion_blur", "color_distort", "overlay_occluder"]
FAMILY = {t: ("T" if i < 6 else "H") for i, t in enumerate(TYPE_ORDER)}
SEVS = [1, 3, 5]


def mean_std(df, col):
    g = df.groupby("seed")[col].mean() if "seed" in df else df[col]
    return float(g.mean()), (float(g.std(ddof=1)) if len(g) > 1 else 0.0)


def fig1(rob, arms, out):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True)
    for ax, t in zip(axes.ravel(), TYPE_ORDER):
        for a in arms:
            xs, ms, ss = [], [], []
            for s in SEVS:
                d = rob[(rob.arm == a) & (rob.type == t) & (rob.severity == s)]
                if len(d):
                    m, sd = mean_std(d, "auc")
                    xs.append(s); ms.append(m); ss.append(sd)
            if xs:
                ms, ss = np.array(ms), np.array(ss)
                ax.plot(xs, ms, "-o", color=COLORS[a], label=NAMES[a], lw=2)
                ax.fill_between(xs, ms - ss, ms + ss, color=COLORS[a], alpha=0.2)
        ax.set_title(f"[{FAMILY[t]}] {t}", fontsize=10)
        ax.set_xticks(SEVS)
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(TYPE_ORDER):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")
    fig.suptitle("AUC vs corruption severity ([T] = training family, [H] = held out)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def per_run_family(rob, fam, sev, arm):
    d = rob[(rob.arm == arm) & (rob.family == fam) & (rob.severity == sev)]
    return d.groupby("seed")["auc"].mean()


def fig2(rob, arms, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = [(f, s) for f in ("T", "H") for s in SEVS]
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        vals, errs = [], []
        for f, s in cats:
            g = per_run_family(rob, f, s, a)
            vals.append(g.mean() if len(g) else np.nan)
            errs.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        ax.bar(np.arange(len(cats)) + i * w - 0.4 + w / 2, vals, w, yerr=errs, color=COLORS[a], label=NAMES[a], capsize=3)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{f}  sev {s}" for f, s in cats])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("mean AUC over corruption types")
    ax.set_title("Training family (T) vs held-out family (H)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3(rob, arms, out, severity=3):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    w = 0.8 / len(arms)
    for ax, col, title in zip(axes, ("sn", "sp"), ("Sensitivity (real recall)", "Specificity (fake recall)")):
        for i, a in enumerate(arms):
            vals = []
            for t in TYPE_ORDER:
                d = rob[(rob.arm == a) & (rob.type == t) & (rob.severity == severity)]
                vals.append(100 * d[col].mean() if len(d) else np.nan)
            ax.bar(np.arange(len(TYPE_ORDER)) + i * w - 0.4 + w / 2, vals, w, color=COLORS[a], label=NAMES[a])
        ax.set_ylabel(title + " %")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=len(arms), loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    axes[1].set_xticks(range(len(TYPE_ORDER)))
    axes[1].set_xticklabels([f"[{FAMILY[t]}] {t}" for t in TYPE_ORDER], rotation=35, ha="right")
    fig.suptitle(f"Where do errors go? Real vs fake recall at severity {severity}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4(off, arms, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    d = off[off.subset == "all"]
    for ax, col, title in zip(axes, ("acc", "sn", "sp"), ("Accuracy %", "Sensitivity %", "Specificity %")):
        for i, a in enumerate(arms):
            v = 100 * d[d.arm == a][col].values
            if len(v):
                ax.bar(i, v.mean(), color=COLORS[a], alpha=0.8)
                ax.scatter(np.full(len(v), i), v, color="black", s=14, zorder=3)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms)
        ax.set_title(title)
        lo = 100 * d[col].min()
        ax.set_ylim(max(0, lo - 8), 100)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Official test set (dots = individual seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig5(rob, arms, out):
    groups = [("clean pool", lambda d: d[d.family == "clean"]),
              ("T sev 3", lambda d: d[(d.family == "T") & (d.severity == 3)]),
              ("H sev 3", lambda d: d[(d.family == "H") & (d.severity == 3)]),
              ("H sev 5", lambda d: d[(d.family == "H") & (d.severity == 5)])]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        vals = []
        for _, f in groups:
            d = f(rob[rob.arm == a])
            vals.append(d.groupby("seed")["ece"].mean().mean() if len(d) else np.nan)
        ax.bar(np.arange(len(groups)) + i * w - 0.4 + w / 2, vals, w, color=COLORS[a], label=NAMES[a])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("ECE (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Calibration under corruption")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/study_results")
    ap.add_argument("--severity", type=int, default=3, help="severity used in figure 3")
    args = ap.parse_args()
    rob = pd.read_csv(os.path.join(args.dir, "robust_by_condition_per_run.csv"))
    off = pd.read_csv(os.path.join(args.dir, "official_test_per_run.csv"))
    order = ["A", "B", "BN", "CnoNJ", "C"]
    arms = [a for a in order if a in set(rob.arm)] + sorted(set(rob.arm) - set(order))
    for a in arms:
        COLORS.setdefault(a, "#2ca02c")
        NAMES.setdefault(a, a)
    fig1(rob, arms, os.path.join(args.dir, "fig1_auc_vs_severity.png"))
    fig2(rob, arms, os.path.join(args.dir, "fig2_family_severity.png"))
    fig3(rob, arms, os.path.join(args.dir, "fig3_sn_sp_by_type.png"), args.severity)
    fig4(off, arms, os.path.join(args.dir, "fig4_official_test.png"))
    fig5(rob, arms, os.path.join(args.dir, "fig5_calibration.png"))
    print(f"saved 5 figures in {args.dir}")


if __name__ == "__main__":
    main()
