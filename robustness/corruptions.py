"""
corruptions.py

Deterministic, label-free image corruptions for the multi-axis robustness study.
Only numpy + Pillow are needed (no torch), so this file can be tested anywhere:

    python -m robustness.corruptions        # runs a self-test

Two families (never mix them up in a paper):
  T  (train family)    used for training arm C and for synthetic validation
        lighting, color_cast, gaussian_blur, block_occlusion, gaussian_noise, jpeg
  H  (held-out family) NEVER used in training; only for evaluating generalisation
        salt_pepper, pixelate, motion_blur, color_distort, overlay_occluder

Every corruption has 5 severities (1 = mild ... 5 = severe) and takes
    f(img: PIL.Image RGB, severity: int, rng: np.random.Generator) -> PIL.Image RGB

Design rules that matter for the science:
  * Nothing here ever sees a label. The sampler only receives (seed, epoch, index).
  * Everything is reproducible: a "plan" is a list of {type, severity, seed}; applying the same
    plan to the same image always gives the same pixels.
  * Occluders are SYNTHETIC overlays placed at random (images are not face-aligned), so call
    them "synthetic occluders", never "real-world occlusion".
"""

import io
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TRAIN_FAMILY = ["lighting", "color_cast", "gaussian_blur", "block_occlusion", "gaussian_noise", "jpeg"]
HELDOUT_FAMILY = ["salt_pepper", "pixelate", "motion_blur", "color_distort", "overlay_occluder"]
ALL_TYPES = TRAIN_FAMILY + HELDOUT_FAMILY          # canonical order, also the composition order
FAMILIES = {"T": TRAIN_FAMILY, "H": HELDOUT_FAMILY, "TH": ALL_TYPES}
SEVERITIES = (1, 2, 3, 4, 5)

# ---- severity tables (tune ONLY on synthetic validation, never on the official test set) ----
_NOISE_SIGMA = {1: 8, 2: 16, 3: 24, 4: 36, 5: 50}              # gaussian_noise, 0-255 scale
_BLUR_SIGMA = {1: 0.8, 2: 1.4, 3: 2.0, 4: 2.8, 5: 3.6}          # gaussian_blur
_JPEG_Q = {1: 70, 2: 50, 3: 35, 4: 20, 5: 10}                  # jpeg quality
_AREA = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}          # occluder area / image area
_GAMMA = {1: 1.2, 2: 1.5, 3: 1.8, 4: 2.1, 5: 2.5}              # lighting
_SCALE = {1: 0.9, 2: 0.75, 3: 0.6, 4: 0.5, 5: 0.4}
_CONTRAST = {1: 0.95, 2: 0.85, 3: 0.75, 4: 0.65, 5: 0.55}
_GAIN = {1: 1.03, 2: 1.07, 3: 1.12, 4: 1.17, 5: 1.22}
_BGAMMA = {1: 1 / 1.1, 2: 1 / 1.25, 3: 1 / 1.4, 4: 1 / 1.6, 5: 1 / 1.8}   # over-exposure branch
_CAST = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}               # color_cast strength
_SP_FRAC = {1: 0.01, 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12}       # salt_pepper pixel fraction
_PIX = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8}                          # pixelate factor
_MOTION = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}                    # motion_blur kernel length (px)
_DISTORT = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}            # color_distort blend


def _to_f(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)       # float32 conversion => writable copy


def _to_img(a):
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ family T
def gaussian_noise(img, severity, rng):
    a = _to_f(img)
    return _to_img(a + rng.normal(0.0, _NOISE_SIGMA[severity], a.shape).astype(np.float32))


def lighting(img, severity, rng):
    """Dull / low light (50%) or harsh over-exposure (50%), plus reduced contrast."""
    a = _to_f(img) / 255.0
    g, s, c = _GAMMA[severity], _SCALE[severity], _CONTRAST[severity]
    if rng.random() < 0.5:
        out = (a ** g) * s
    else:
        out = np.clip((a ** _BGAMMA[severity]) * _GAIN[severity], 0.0, 1.0)
    m = out.mean()
    out = (out - m) * c + m
    return _to_img(out * 255.0)


def color_cast(img, severity, rng):
    a = _to_f(img)
    d = _CAST[severity]
    gains = (1.0 + rng.uniform(-d, d, size=3)).astype(np.float32)
    sat = float(1.0 + rng.uniform(-d, d))
    gray = a.mean(axis=2, keepdims=True)
    out = (gray + (a - gray) * sat) * gains.reshape(1, 1, 3)
    return _to_img(out)


def gaussian_blur(img, severity, rng):
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=_BLUR_SIGMA[severity]))


def jpeg(img, severity, rng):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=_JPEG_Q[severity])
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def _rect_dims(rng, area_frac, W, H, lo=0.5, hi=2.0):
    area = area_frac * W * H
    r = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    w = int(round(min(W, np.sqrt(area * r))))
    h = int(round(min(H, area / max(w, 1))))
    return max(w, 1), max(h, 1)


def block_occlusion(img, severity, rng):
    """1-2 axis-aligned rectangles (black / mean colour / random colour / noise patch)."""
    a = _to_f(img)
    H, W = a.shape[:2]
    n = 1 if severity <= 3 else 2
    for _ in range(n):
        w, h = _rect_dims(rng, _AREA[severity] / n, W, H)
        x0 = int(rng.integers(0, W - w + 1))
        y0 = int(rng.integers(0, H - h + 1))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            fill = 0.0
        elif mode == 1:
            fill = a.mean(axis=(0, 1))
        elif mode == 2:
            fill = rng.uniform(0, 255, size=3).astype(np.float32)
        else:
            fill = rng.uniform(0, 255, size=(h, w, 3)).astype(np.float32)
        a[y0:y0 + h, x0:x0 + w, :] = fill
    return _to_img(a)


# ------------------------------------------------------------------ family H (held out)
def salt_pepper(img, severity, rng):
    """'Snow': mostly white specks plus some black ones, same value on all channels."""
    a = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    H, W = a.shape[:2]
    hit = rng.random((H, W)) < _SP_FRAC[severity]
    salt = rng.random((H, W)) < 0.7
    a[hit & salt] = 255
    a[hit & ~salt] = 0
    return Image.fromarray(a)


def pixelate(img, severity, rng):
    img = img.convert("RGB")
    W, H = img.size
    f = _PIX[severity]
    small = img.resize((max(1, W // f), max(1, H // f)), Image.BILINEAR)
    return small.resize((W, H), Image.NEAREST)


def motion_blur(img, severity, rng):
    a = _to_f(img)
    H, W = a.shape[:2]
    L = _MOTION[severity]
    theta = float(rng.choice([0.0, 45.0, 90.0, 135.0]) + rng.uniform(-10.0, 10.0)) * np.pi / 180.0
    pad = L // 2 + 1
    p = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(a)
    for t in np.linspace(-(L - 1) / 2.0, (L - 1) / 2.0, L):
        dx = int(round(t * np.cos(theta)))
        dy = int(round(t * np.sin(theta)))
        acc += p[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]
    return _to_img(acc / L)


def color_distort(img, severity, rng):
    """Blend towards colour inversion or a channel permutation."""
    a = _to_f(img)
    al = _DISTORT[severity]
    if rng.random() < 0.5:
        target = 255.0 - a
    else:
        perm = rng.permutation(3)
        while (perm == np.arange(3)).all():
            perm = rng.permutation(3)
        target = a[:, :, perm]
    return _to_img((1.0 - al) * a + al * target)


def overlay_occluder(img, severity, rng):
    """Textured synthetic occluders with different shapes from block_occlusion:
    ellipse, 'sunglasses-like' dark band, or 'mask-like' rounded patch. Random position biased
    to the central region (images are not face-aligned, so this is only approximate)."""
    base = img.convert("RGB")
    W, H = base.size
    area = _AREA[severity] * W * H
    kind = ("ellipse", "band", "mask")[int(rng.integers(0, 3))]
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    if kind == "ellipse":
        r = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6))))
        ra, rb = np.sqrt(area * r / np.pi), np.sqrt(area / (np.pi * r))
        cx, cy = rng.uniform(0.25 * W, 0.75 * W), rng.uniform(0.25 * H, 0.75 * H)
        d.ellipse([cx - ra, cy - rb, cx + ra, cy + rb], fill=255)
        col = tuple(int(v) for v in rng.integers(30, 226, size=3))
    elif kind == "band":
        w = min(float(W), float(np.sqrt(4.0 * area)))
        h = area / w
        cx, cy = rng.uniform(0.35 * W, 0.65 * W), rng.uniform(0.30 * H, 0.50 * H)
        d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=max(1, int(h / 3)), fill=255)
        col = (int(rng.integers(5, 40)),) * 3
    else:
        w = min(float(W), float(np.sqrt(1.6 * area)))
        h = area / w
        cx, cy = rng.uniform(0.35 * W, 0.65 * W), rng.uniform(0.55 * H, 0.80 * H)
        d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=max(1, int(h / 4)), fill=255)
        palette = [(200, 225, 240), (235, 235, 235), (150, 190, 210), (90, 100, 110)]
        col = palette[int(rng.integers(0, len(palette)))]
    tex = np.empty((H, W, 3), dtype=np.float32)
    tex[:] = col
    period = int(rng.integers(3, 8))
    stripes = ((np.arange(H)[:, None] // period) % 2).astype(np.float32) * 0.15
    tex *= (1.0 - stripes)[:, :, None]
    layer = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))
    out = base.copy()
    out.paste(layer, (0, 0), mask)
    return out


_FUNCS = {
    "gaussian_noise": gaussian_noise, "lighting": lighting, "color_cast": color_cast,
    "gaussian_blur": gaussian_blur, "jpeg": jpeg, "block_occlusion": block_occlusion,
    "salt_pepper": salt_pepper, "pixelate": pixelate, "motion_blur": motion_blur,
    "color_distort": color_distort, "overlay_occluder": overlay_occluder,
}


def family_of(ctype):
    return "T" if ctype in TRAIN_FAMILY else ("H" if ctype in HELDOUT_FAMILY else "clean")


def apply_corruption(img, ctype, severity, rng):
    if ctype not in _FUNCS:
        raise ValueError(f"unknown corruption '{ctype}'")
    if int(severity) not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    out = _FUNCS[ctype](img, int(severity), rng)
    if out.size != img.size:
        raise RuntimeError(f"{ctype} changed the image size")
    return out


def apply_plan(img, plan):
    """plan = [{'type':..., 'severity':..., 'seed':...}, ...]; deterministic."""
    for op in plan:
        rng = np.random.default_rng(int(op["seed"]))
        img = apply_corruption(img, op["type"], int(op["severity"]), rng)
    return img


def describe_plan(plan):
    return "+".join(f"{op['type']}@{op['severity']}" for op in plan) if plan else "clean"


# ------------------------------------------------------------------ samplers
class StructuredCorruptor:
    """Label-independent training-time sampler.

    With probability p an image is corrupted by 1 (or, with prob two_op_prob, 2) corruption(s)
    drawn uniformly from `family`, each with severity drawn uniformly from `severities`.
    Real and fake images are treated identically because the sampler never sees the label:
    the plan depends only on (seed, epoch, index)."""

    def __init__(self, p=0.5, family="T", max_ops=2, two_op_prob=0.3, severities=SEVERITIES):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {list(FAMILIES)}")
        self.p = float(p)
        self.types = list(FAMILIES[family])
        self.max_ops = int(max_ops)
        self.two_op_prob = float(two_op_prob)
        self.severities = tuple(severities)

    def sample_plan(self, rng):
        if rng.random() >= self.p:
            return []
        n = 2 if (self.max_ops >= 2 and rng.random() < self.two_op_prob) else 1
        types = [str(t) for t in rng.choice(self.types, size=n, replace=False)]
        order = {t: i for i, t in enumerate(ALL_TYPES)}
        types.sort(key=lambda t: order[t])                      # canonical composition order
        return [{"type": t,
                 "severity": int(rng.choice(self.severities)),
                 "seed": int(rng.integers(0, 2 ** 31 - 1))} for t in types]

    def plan_for(self, seed, epoch, index):
        """THE ONLY place plans are generated for training, so audits test the real code path."""
        rng = np.random.default_rng([int(seed), int(epoch), int(index), 7])
        return self.sample_plan(rng)


def condition_plan(ctype, severity, base_seed, index):
    """Plan for one evaluation condition applied to pool image `index` (identical for every model)."""
    cond_id = zlib.crc32(f"{ctype}:{int(severity)}".encode())
    rng = np.random.default_rng([int(base_seed), int(cond_id), int(index)])
    return [{"type": ctype, "severity": int(severity), "seed": int(rng.integers(0, 2 ** 31 - 1))}]


# ------------------------------------------------------------------ self-test
def _selftest():
    rng0 = np.random.default_rng(0)
    base = rng0.normal(128, 40, (256, 256, 3)).clip(0, 255).astype(np.uint8)
    base = np.asarray(Image.fromarray(base).filter(ImageFilter.GaussianBlur(1.5)))   # textured, photo-like
    img = Image.fromarray(base)
    before = np.asarray(img).copy()
    for t in ALL_TYPES:
        diffs = {}
        for s in SEVERITIES:
            ds = []
            for seed in range(6):
                out = apply_corruption(img, t, s, np.random.default_rng(seed))
                assert out.size == (256, 256) and out.mode == "RGB", t
                ds.append(np.abs(np.asarray(out).astype(np.float32) - before.astype(np.float32)).mean())
            a1 = apply_corruption(img, t, s, np.random.default_rng(123))
            a2 = apply_corruption(img, t, s, np.random.default_rng(123))
            assert (np.asarray(a1) == np.asarray(a2)).all(), f"{t} not deterministic"
            diffs[s] = float(np.mean(ds))
        assert diffs[5] > diffs[1], f"{t}: severity 5 should change the image more than severity 1 {diffs}"
        print(f"  ok {t:<17} mean|diff| sev1={diffs[1]:6.2f}  sev5={diffs[5]:6.2f}")
    assert (np.asarray(img) == before).all(), "input image was modified in place"

    cor = StructuredCorruptor(p=0.5)
    plans = [cor.plan_for(0, 1, i) for i in range(20000)]
    frac = np.mean([len(p) > 0 for p in plans])
    assert 0.48 < frac < 0.52, frac
    assert plans == [cor.plan_for(0, 1, i) for i in range(20000)], "plans not reproducible"
    assert all(op["type"] in TRAIN_FAMILY for p in plans for op in p), "held-out type leaked into training sampler"
    assert condition_plan("jpeg", 3, 0, 5) == condition_plan("jpeg", 3, 0, 5)
    print(f"  ok sampler: corrupted fraction={frac:.3f}, reproducible, train-family only")
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
