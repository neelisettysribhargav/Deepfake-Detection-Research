"""
scripts/load_cswin.py

Loads the OFFICIAL ImageNet-1K CSWin-S (224x224, 35M params) checkpoint from the Microsoft
CSWin-Transformer repo into this project's CSWin_64_24322_small_256 model.

Why this should work without resizing:
  * CSWin uses LePE (a depth-wise conv on V) instead of a position table, so no weight depends
    on the input resolution. 224 -> 256 changes only the window size (7 -> 8), not any shape.
  * Key names match the official model except the patch embedding (official nn.Sequential
    indices 0 and 2 -> this project's PatchEmbed.proj / PatchEmbed.norm). This script remaps
    those two and skips the 1000-class head.

Nothing is assumed silently: the script prints how many tensors loaded, which model tensors
were NOT initialised from the checkpoint, and any shape mismatches. The only model tensors
that should remain un-loaded are head.weight and head.bias.

Usage (from the project root):
  python scripts/load_cswin.py --weights path/to/cswin_small_224.pth
  python scripts/load_cswin.py --weights path/to/cswin_small_224.pth --out cswin_s_imagenet_init.pth

Only load checkpoint files you downloaded from the official repo (older torch.load falls back
to pickle for non-tensor entries).
"""

import os
import re
import sys
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.dirname(HERE))


def remap_official_keys(state):
    out = {}
    for k, v in state.items():
        k = re.sub(r"^module\.", "", k)
        k = k.replace("stage1_conv_embed.0.", "stage1_conv_embed.proj.")
        k = k.replace("stage1_conv_embed.2.", "stage1_conv_embed.norm.")
        out[k] = v
    return out


def _read_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        print("[warn] weights_only load failed; falling back to full pickle load. "
              "Only do this for the official file.")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_official_cswin_weights(model, ckpt_path, min_loaded_fraction=0.95, verbose=True):
    ckpt = _read_checkpoint(ckpt_path)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict_ema", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    state = remap_official_keys(state)

    model_sd = model.state_dict()
    to_load, shape_mismatch, unexpected = {}, [], []
    for k, v in state.items():
        if k not in model_sd:
            unexpected.append(k)
        elif model_sd[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            to_load[k] = v

    model.load_state_dict(to_load, strict=False)
    not_loaded = [k for k in model_sd if k not in to_load]
    frac = len(to_load) / max(1, len(model_sd))

    if verbose:
        print(f"Loaded {len(to_load)}/{len(model_sd)} model tensors ({frac * 100:.1f}%)")
        print(f"Model tensors NOT initialised from checkpoint ({len(not_loaded)}): "
              f"{not_loaded[:12]}{' ...' if len(not_loaded) > 12 else ''}")
        print(f"Checkpoint tensors with no matching model key ({len(unexpected)}): "
              f"{unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")
        print(f"Shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:8]}")

    head_only = all(k.startswith("head.") for k in not_loaded)
    if frac < min_loaded_fraction or not head_only:
        raise RuntimeError(
            "Pretrained weights did not map cleanly (expected only head.* to be left out). "
            "Send this printout so the key mapping can be fixed; do not train on a half-loaded model."
        )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="official CSWin-S 224 ImageNet-1K checkpoint")
    ap.add_argument("--out", default=None, help="optional: save the initialised model state_dict")
    args = ap.parse_args()

    from models.model_factory import get_model

    model = get_model("CSWin_64_24322_small_256", num_classes=1, pretrained=False)
    load_official_cswin_weights(model, args.weights)
    print("OK: backbone initialised from ImageNet weights; head.* is freshly initialised.")
    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
