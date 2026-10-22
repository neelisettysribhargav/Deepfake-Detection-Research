"""
aggregate_study.py

Turns the per-run prediction files into the paper's tables and statistics.

Inputs (written by evaluate.py / evaluate_robustness.py for every run `cswin_<arm>_s<seed>`):
  experiments/predictions/<tag>_predictions.csv          official val/test per-image logits
  experiments/predictions/<tag>_metrics.csv              (used for the clean-validation gate)
  experiments/robustness/<tag>_robust_predictions.csv    per-image logits for every condition

Outputs in --out (default experiments/study_results):
  official_test_per_run.csv, official_test_summary.csv   mean +/- std over seeds
  robust_by_condition_per_run.csv, robust_by_condition.csv, robust_family_severity.csv
  paired_bootstrap.csv                                   differences between arms with 95% CIs
  summary.md                                             tables ready to paste, plus gate verdicts

Statistics:
  * seed variability: mean and std over seeds (reported, not bootstrapped);
  * image variability: paired bootstrap, the SAME resampled images for every arm and seed
    (official test and synthetic pool: over images; DFDC: over videos, because frames of one
    video are correlated). Statistic per arm = mean over its seeds. 95% percentile CI.
  * AUC is the primary metric. Sn (real recall) and Sp (fake recall) are reported separately.

  python scripts/aggregate_study.py --arms A B C --seeds 1 2 3 --n-boot 500
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

POOL_SEVS = (1, 3, 5)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def auc_fast(y, s):
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece_brier(prob, y, n_bins=15):
    pred = (prob >= 0.5).astype(int)
    conf = np.maximum(prob, 1.0 - prob)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece), float(np.mean((prob - y) ** 2))


def metrics(y, z):
    prob = sig(z)
    pred = (prob >= 0.5).astype(int)
    sn = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    sp = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    ece, brier = ece_brier(prob, y)
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "bal_acc": (sn + sp) / 2, "auc": auc_fast(y, z),
            "sn": sn, "sp": sp, "ece": ece, "brier": brier}


def read_csv(path):
    try:
        d = pd.read_csv(path, skipinitialspace=True)
    except Exception as e:
        raise SystemExit(f"cannot parse {path}: {e}\nRun scripts/check_prediction_files.py and re-run the evaluation for the bad tag.")
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype(str).str.strip()
    return d


def video_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def ms(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--prefix", default="cswin")
    ap.add_argument("--pred-dir", default="experiments/predictions")
    ap.add_argument("--rob-dir", default="experiments/robustness")
    ap.add_argument("--out", default="experiments/study_results")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---------------- load
    OFF, ROB, VAL = {}, {}, {}
    seeds_by_arm = {}
    for arm in args.arms:
        seeds_by_arm[arm] = []
        for s in args.seeds:
            tag = f"{args.prefix}_{arm}_s{s}"
            pp = os.path.join(args.pred_dir, f"{tag}_predictions.csv")
            rp = os.path.join(args.rob_dir, f"{tag}_robust_predictions.csv")
            mp = os.path.join(args.pred_dir, f"{tag}_metrics.csv")
            if not (os.path.isfile(pp) and os.path.isfile(rp)):
                print(f"[skip] {tag}: prediction files not found yet")
                continue
            d = read_csv(pp)
            t = d[d.split == "test"].sort_values("path").reset_index(drop=True)
            OFF[(arm, s)] = t
            r = read_csv(rp)
            ROB[(arm, s)] = {c: g.sort_values("path").reset_index(drop=True) for c, g in r.groupby("condition")}
            if os.path.isfile(mp):
                m = read_csv(mp).set_index("group")
                if "validation/dedup" in m.index:
                    VAL[(arm, s)] = float(m.loc["validation/dedup", "acc"])
            seeds_by_arm[arm].append(s)
    arms = [a for a in args.arms if seeds_by_arm[a]]
    if len(arms) < 2:
        raise SystemExit("need at least two arms with finished runs")
    print("runs found:", {a: seeds_by_arm[a] for a in arms})

    # alignment check
    ref_arm, ref_s = arms[0], seeds_by_arm[arms[0]][0]
    ref_paths = OFF[(ref_arm, ref_s)].path.values
    for k, t in OFF.items():
        if not np.array_equal(t.path.values, ref_paths):
            raise SystemExit(f"official test images of {k} differ from the reference run")
    conditions = sorted(ROB[(ref_arm, ref_s)].keys())
    for k, rr in ROB.items():
        for c in conditions:
            if c not in rr or not np.array_equal(rr[c].path.values, ROB[(ref_arm, ref_s)][c].path.values):
                raise SystemExit(f"robustness images of {k} / {c} differ from the reference run")

    # ---------------- per-run tables
    off_rows, rob_rows = [], []
    for (arm, s), t in OFF.items():
        y, z = t.label.values.astype(int), t.logit.values
        off_rows.append({"arm": arm, "seed": s, "subset": "all", **metrics(y, z)})
        for sub, g in t.groupby("subset"):
            off_rows.append({"arm": arm, "seed": s, "subset": sub, **metrics(g.label.values.astype(int), g.logit.values)})
    for (arm, s), rr in ROB.items():
        for c, g in rr.items():
            meta = g.iloc[0]
            rob_rows.append({"arm": arm, "seed": s, "condition": c, "family": meta.family, "type": meta.type,
                             "severity": int(meta.severity), **metrics(g.label.values.astype(int), g.logit.values)})
    off = pd.DataFrame(off_rows)
    rob = pd.DataFrame(rob_rows)
    off.to_csv(os.path.join(args.out, "official_test_per_run.csv"), index=False)
    rob.to_csv(os.path.join(args.out, "robust_by_condition_per_run.csv"), index=False)

    def summarize(df, keys, cols):
        rows = []
        for k, g in df.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            row = dict(zip(keys, k))
            row["n_seeds"] = g.seed.nunique() if "seed" in g else len(g)
            for c in cols:
                m_, s_ = ms(g[c])
                row[f"{c}_mean"], row[f"{c}_std"] = m_, s_
            rows.append(row)
        return pd.DataFrame(rows)

    cols = ["acc", "bal_acc", "auc", "sn", "sp", "ece"]
    off_sum = summarize(off, ["arm", "subset"], cols)
    off_sum.to_csv(os.path.join(args.out, "official_test_summary.csv"), index=False)
    rob_sum = summarize(rob, ["arm", "condition", "family", "type", "severity"], cols)
    rob_sum.to_csv(os.path.join(args.out, "robust_by_condition.csv"), index=False)

    # family x severity: mean over types within run, then mean/std over seeds
    fs_rows = []
    for (arm, s, fam, sev), g in rob[rob.family.isin(["T", "H"])].groupby(["arm", "seed", "family", "severity"]):
        fs_rows.append({"arm": arm, "seed": s, "family": fam, "severity": sev, **{c: g[c].mean() for c in cols}})
    fs = summarize(pd.DataFrame(fs_rows), ["arm", "family", "severity"], cols)
    fs.to_csv(os.path.join(args.out, "robust_family_severity.csv"), index=False)

    # ---------------- paired bootstrap
    comps = [(a, b) for a in arms for b in arms if a > b]          # e.g. B-A, C-A, C-B
    comps = sorted(comps, key=lambda ab: (ab[0], ab[1]))
    n_off = len(ref_paths)
    pool_n = len(ROB[(ref_arm, ref_s)]["pool"])
    pool_types = {c: (ROB[(ref_arm, ref_s)][c].iloc[0].family, int(ROB[(ref_arm, ref_s)][c].iloc[0].severity)) for c in conditions}
    groups = {"clean": ["pool"]}
    for fam in ("T", "H"):
        for sev in POOL_SEVS:
            cs = [c for c, (f, sv) in pool_types.items() if f == fam and sv == sev]
            if cs:
                groups[f"{fam}{sev}"] = cs
    has_dfdc = "dfdc_clean" in conditions
    if has_dfdc:
        dpaths = ROB[(ref_arm, ref_s)]["dfdc_clean"].path.values
        vids = pd.Series([video_of(p) for p in dpaths])
        vid_idx = [np.where(vids.values == v)[0] for v in vids.unique()]

    def arm_stats(arm, idx_off, idx_pool, idx_dfdc):
        out = {}
        # official test
        a_, u_, n_ = [], [], []
        for s in seeds_by_arm[arm]:
            t = OFF[(arm, s)]
            y, z = t.label.values.astype(int)[idx_off], t.logit.values[idx_off]
            pred = z >= 0
            u_.append(auc_fast(y, z)); a_.append((pred == y).mean()); n_.append(pred[y == 1].mean())
        out[("official", "auc")], out[("official", "acc")], out[("official", "sn")] = np.mean(u_), np.mean(a_), np.mean(n_)
        # synthetic pool
        for gname, cs in groups.items():
            v = []
            for s in seeds_by_arm[arm]:
                for c in cs:
                    g = ROB[(arm, s)][c]
                    v.append(auc_fast(g.label.values.astype(int)[idx_pool], g.logit.values[idx_pool]))
            out[(f"pool_{gname}", "auc")] = np.mean(v)
        if has_dfdc:
            v = []
            for s in seeds_by_arm[arm]:
                g = ROB[(arm, s)]["dfdc_clean"]
                v.append(auc_fast(g.label.values.astype(int)[idx_dfdc], g.logit.values[idx_dfdc]))
            out[("dfdc", "auc")] = np.mean(v)
        return out

    all_off, all_pool = np.arange(n_off), np.arange(pool_n)
    all_dfdc = np.arange(len(dpaths)) if has_dfdc else None
    point = {a: arm_stats(a, all_off, all_pool, all_dfdc) for a in arms}
    draws = {a: [] for a in arms}
    for _ in range(args.n_boot):
        io = rng.integers(0, n_off, n_off)
        ip = rng.integers(0, pool_n, pool_n)
        idf = np.concatenate([vid_idx[j] for j in rng.integers(0, len(vid_idx), len(vid_idx))]) if has_dfdc else None
        for a in arms:
            draws[a].append(arm_stats(a, io, ip, idf))
    rows = []
    for key in point[arms[0]].keys():
        for a, b in comps:
            d = np.array([draws[a][i][key] - draws[b][i][key] for i in range(args.n_boot)])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            diff = point[a][key] - point[b][key]
            rows.append({"dataset": key[0], "metric": key[1], "comparison": f"{a}-{b}", "diff": diff, "ci_lo": lo, "ci_hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    pb = pd.DataFrame(rows)
    pb.to_csv(os.path.join(args.out, "paired_bootstrap.csv"), index=False)

    # ---------------- summary.md + gates
    L = ["# Study summary", "", f"Runs: { {a: seeds_by_arm[a] for a in arms} }, bootstrap resamples: {args.n_boot}", ""]
    L += ["## Official test (all subsets), mean +/- std over seeds", "", "| Arm | Acc % | AUC | Sn % | Sp % |", "|---|---|---|---|---|"]
    for a in arms:
        r = off_sum[(off_sum.arm == a) & (off_sum.subset == "all")].iloc[0]
        L.append(f"| {a} | {100 * r.acc_mean:.2f} +/- {100 * r.acc_std:.2f} | {r.auc_mean:.4f} +/- {r.auc_std:.4f} | "
                 f"{100 * r.sn_mean:.1f} +/- {100 * r.sn_std:.1f} | {100 * r.sp_mean:.1f} +/- {100 * r.sp_std:.1f} |")
    L += ["", "## Synthetic pool: mean AUC over corruption types, mean +/- std over seeds", "",
          "| Family | Severity | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for fam in ("T", "H"):
        for sev in POOL_SEVS:
            cells = []
            for a in arms:
                r = fs[(fs.arm == a) & (fs.family == fam) & (fs.severity == sev)]
                cells.append(f"{r.auc_mean.iloc[0]:.4f} +/- {r.auc_std.iloc[0]:.4f}" if len(r) else "-")
            L.append(f"| {fam} | {sev} | " + " | ".join(cells) + " |")
    L += ["", "## Paired bootstrap differences (95% CI)", "", "| Dataset | Metric | Comparison | Diff | 95% CI | CI excludes 0 |", "|---|---|---|---|---|---|"]
    for _, r in pb.iterrows():
        if r.dataset in ("official", "pool_H3", "pool_H5", "pool_T3", "pool_clean", "dfdc"):
            L.append(f"| {r.dataset} | {r.metric} | {r.comparison} | {r['diff']:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {'yes' if r.excludes_zero else 'no'} |")

    L += ["", "## Pre-specified gates (arm C)", ""]
    if "C" in arms and "B" in arms:
        g1 = pb[(pb.dataset == "pool_H3") & (pb.comparison == "C-B")].iloc[0]
        L.append(f"1. C beats B on held-out family H, mean AUC at severity 3: diff {g1['diff']:+.4f} "
                 f"[{g1.ci_lo:+.4f}, {g1.ci_hi:+.4f}] -> {'PASS' if g1.ci_lo > 0 else 'FAIL'}")
    if "C" in arms and "A" in arms and all((a, s) in VAL for a in ("A", "C") for s in seeds_by_arm[a]):
        va = np.mean([VAL[("A", s)] for s in seeds_by_arm["A"]])
        vc = np.mean([VAL[("C", s)] for s in seeds_by_arm["C"]])
        L.append(f"2. Clean dedup-validation accuracy of C within 1 point of A: A {100 * va:.2f}%, C {100 * vc:.2f}% "
                 f"-> {'PASS' if vc >= va - 0.01 else 'FAIL'}")
    if has_dfdc:
        L.append("3. DFDC is reported as a secondary result (baseline AUC is near chance).")
    text = "\n".join(L)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved results in {args.out}")


if __name__ == "__main__":
    main()
