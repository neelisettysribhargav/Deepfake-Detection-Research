"""
create_author_dataset.py

Option A: Manifest-Defined Benchmark Dataset
Generates the frozen master manifest for the author's official 190,335 benchmark
images (starting with 'real_' and 'fake_') from data/Dataset-1.

Produces:
  1. data/manifests/author_data.csv:
     - path (relative, normalized with forward slashes)
     - subset (Data Set 1, Data Set 2, Data Set 3, Data Set 4)
     - split (train, validation, test)
     - class_name (fake, real)
     - label (0: fake, 1: real)
     - sha256 (unique cryptographic hash for integrity)

  2. data/manifests/author_data_summary.csv:
     - Per-split, per-subset, per-class count table verifying exact
       parity with Table 1 of the paper (140,002 train, 39,428 val, 10,905 test).

Optional:
  --copy-to <dir>: Generates physical folder view if desired, but default is
                   manifest-based to save disk space and guarantee immutability.
"""

import os
import sys
import csv
import shutil
import hashlib
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def process_file_entry(item):
    src_path, subset, split, class_name, label = item
    file_hash = compute_sha256(src_path)
    # Normalize path with forward slashes for cross-platform consistency
    normalized_path = src_path.replace("\\", "/")
    return {
        "path": normalized_path,
        "subset": subset,
        "split": split,
        "class_name": class_name,
        "label": label,
        "sha256": file_hash
    }

def collect_benchmark_files(src_root, subsets, splits, classes):
    items = []
    skipped_count = 0

    for subset in subsets:
        for split in splits:
            for class_name, label in classes.items():
                dir_path = os.path.join(src_root, subset, split, class_name)
                if not os.path.exists(dir_path):
                    continue

                for fname in os.listdir(dir_path):
                    if fname.startswith(("real_", "fake_")):
                        full_path = os.path.join(dir_path, fname)
                        items.append((full_path, subset, split, class_name, label))
                    else:
                        skipped_count += 1

    return items, skipped_count

def verify_and_save_summary(manifest_rows, summary_csv_path):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    split_totals = defaultdict(lambda: defaultdict(int))
    grand_total = 0

    for row in manifest_rows:
        subset = row["subset"]
        split = row["split"]
        cname = row["class_name"]
        counts[subset][split][cname] += 1
        split_totals[split][cname] += 1
        grand_total += 1

    print("\n" + "=" * 78)
    print("AUTHOR BENCHMARK DATASET VERIFICATION (PAPER TABLE 1)")
    print("=" * 78)
    print(f"{'Subset':<12} | {'Split':<12} | {'Fake (0)':<10} | {'Real (1)':<10} | {'Total':<10}")
    print("-" * 78)

    summary_rows = []
    subsets = ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]
    splits = ["train", "validation", "test"]

    for sub in subsets:
        for sp in splits:
            fake_cnt = counts[sub][sp]["fake"]
            real_cnt = counts[sub][sp]["real"]
            tot = fake_cnt + real_cnt
            print(f"{sub:<12} | {sp:<12} | {fake_cnt:<10,d} | {real_cnt:<10,d} | {tot:<10,d}")
            summary_rows.append({
                "subset": sub,
                "split": sp,
                "fake": fake_cnt,
                "real": real_cnt,
                "total": tot
            })

    print("-" * 78)
    print("SPLIT TOTALS:")
    for sp in splits:
        f_tot = split_totals[sp]["fake"]
        r_tot = split_totals[sp]["real"]
        tot = f_tot + r_tot
        print(f"{'TOTAL':<12} | {sp:<12} | {f_tot:<10,d} | {r_tot:<10,d} | {tot:<10,d}")
        summary_rows.append({
            "subset": "ALL_SUBSETS",
            "split": sp,
            "fake": f_tot,
            "real": r_tot,
            "total": tot
        })

    print("-" * 78)
    print(f"GRAND TOTAL BENCHMARK IMAGES: {grand_total:,} (Paper Target: 190,335)")
    print("=" * 78)

    # Assert Table 1 Ground Truth
    assert split_totals["train"]["fake"] == 70001, f"Train fake mismatch: {split_totals['train']['fake']} != 70001"
    assert split_totals["train"]["real"] == 70001, f"Train real mismatch: {split_totals['train']['real']} != 70001"
    assert split_totals["train"]["fake"] + split_totals["train"]["real"] == 140002

    assert split_totals["validation"]["fake"] + split_totals["validation"]["real"] == 39428, f"Val total mismatch: {split_totals['validation']}"
    assert split_totals["test"]["fake"] == 5492, f"Test fake mismatch: {split_totals['test']['fake']} != 5492"
    assert split_totals["test"]["real"] == 5413, f"Test real mismatch: {split_totals['test']['real']} != 5413"
    assert split_totals["test"]["fake"] + split_totals["test"]["real"] == 10905

    assert grand_total == 190335, f"Grand total mismatch: {grand_total} != 190335"
    print("[PASS] EXACT MATCH: All sample counts match Table 1 of the paper down to the single image!")

    # Write summary CSV
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subset", "split", "fake", "real", "total"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SAVED] Count verification table saved to: {summary_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Create frozen author benchmark manifest (Option A)")
    parser.add_argument("--src", type=str, default="data/Dataset-1", help="Source dataset root directory")
    parser.add_argument("--manifest-out", type=str, default="data/manifests/author_data.csv", help="Output manifest CSV path")
    parser.add_argument("--summary-out", type=str, default="data/manifests/author_data_summary.csv", help="Output summary table CSV path")
    parser.add_argument("--workers", type=int, default=16, help="Worker threads for SHA-256 computation")
    parser.add_argument("--copy-to", type=str, default=None, help="Optional: copy files to folder view")
    args = parser.parse_args()

    subsets = ["Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"]
    splits = ["train", "validation", "test"]
    classes = {"fake": 0, "real": 1}

    print("=" * 78)
    print("CREATING FROZEN AUTHOR BENCHMARK MANIFEST (OPTION A)")
    print(f"Source Directory : {os.path.abspath(args.src)}")
    print(f"Output Manifest  : {os.path.abspath(args.manifest_out)}")
    print(f"Summary Table    : {os.path.abspath(args.summary_out)}")
    print(f"Worker Threads   : {args.workers}")
    print("=" * 78)

    if not os.path.exists(args.src):
        print(f"[ERROR] Source root '{args.src}' does not exist!")
        sys.exit(1)

    print("\n[Step 1/3] Scanning dataset directories...")
    items, skipped = collect_benchmark_files(args.src, subsets, splits, classes)
    print(f"Benchmark images identified : {len(items):,}")
    print(f"Corrupt / noisy unannotated files excluded : {skipped:,}")

    print("\n[Step 2/3] Computing SHA-256 hashes for dataset integrity...")
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in tqdm(executor.map(process_file_entry, items), total=len(items), desc="Hashing images", ncols=85):
            manifest_rows.append(row)

    # Sort rows deterministically: split, subset, class, path
    manifest_rows.sort(key=lambda x: (x["split"], x["subset"], x["label"], x["path"]))

    print("\n[Step 3/3] Saving manifest and verifying counts...")
    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    fieldnames = ["path", "subset", "split", "class_name", "label", "sha256"]
    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[SAVED] Master manifest saved to: {args.manifest_out} ({len(manifest_rows):,} rows)")

    # Verify and write summary table
    verify_and_save_summary(manifest_rows, args.summary_out)

    # Optional file copy
    if args.copy_to:
        print(f"\n[Optional] Generating physical folder view at: {args.copy_to}...")
        for row in tqdm(manifest_rows, desc="Copying files", ncols=85):
            dest_file = os.path.join(args.copy_to, row["subset"], row["split"], row["class_name"], os.path.basename(row["path"]))
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            if not os.path.exists(dest_file):
                shutil.copy2(row["path"], dest_file)
        print(f"[DONE] Folder view created at: {args.copy_to}")

    print("\n" + "=" * 78)
    print("[SUCCESS] Frozen author dataset manifest successfully established.")
    print("=" * 78)

if __name__ == "__main__":
    main()
