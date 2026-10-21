"""
make_dfdc_manifest.py

Builds a manifest for the cross-dataset folder (data/Dataset-2, DFDC-style frames named
<videoid>_<frame>.png inside fake/ and real/) so it can be used with DeepfakeDataset and the
evaluation scripts. Evaluation only: never train on it in the main experiments.

Frames from the same video are strongly correlated, so keep `video_id`: confidence intervals
should be bootstrapped over videos, not frames.

  python scripts/make_dfdc_manifest.py --root data/Dataset-2 --out data/manifests/dfdc_faces.csv
"""

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Dataset-2")
    ap.add_argument("--out", default="data/manifests/dfdc_faces.csv")
    args = ap.parse_args()

    rows = []
    for cls, label in (("fake", 0), ("real", 1)):
        d = os.path.join(args.root, cls)
        if not os.path.isdir(d):
            raise SystemExit(f"missing folder: {d}")
        for name in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            video = stem.rsplit("_", 1)[0] if "_" in stem else stem
            rows.append({"path": os.path.join(d, name).replace("\\", "/"), "subset": "DFDC", "split": "test",
                         "class_name": cls, "label": label, "video_id": video})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "subset", "split", "class_name", "label", "video_id"])
        w.writeheader()
        w.writerows(rows)

    n_fake = sum(r["label"] == 0 for r in rows)
    n_real = len(rows) - n_fake
    videos = {(r["class_name"], r["video_id"]) for r in rows}
    print(f"saved {args.out}: {len(rows)} frames | fake {n_fake} | real {n_real} | videos {len(videos)}")


if __name__ == "__main__":
    main()
