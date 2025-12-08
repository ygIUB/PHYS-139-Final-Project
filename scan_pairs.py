from __future__ import annotations
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from skimage import io

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


ROOT = Path(os.environ.get("EMP_DIR", "/workspace/cem_mitolab")).resolve()
QA_DIR = Path("/workspace/qa")
IMG_DIR_RE = re.compile(r"images?$", re.I)
MSK_DIR_RE = re.compile(r"masks?$",  re.I)
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

STEM_CLEAN_RE = re.compile(r"([_-](ch|loc|slice|z|t)?\d+|[-_](\d+))$", re.I)



def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def find_sibling(dirpath: Path, pat: re.Pattern) -> Path | None:
    parent = dirpath.parent
    for d in parent.iterdir():
        if d.is_dir() and pat.search(d.name):
            return d
    return None


def normalize_stem(stem: str) -> str:
    s = STEM_CLEAN_RE.sub("", stem)
    return s


def try_mask_path(msk_root: Path, rel: Path) -> Path | None:
    cand = (msk_root / rel.with_suffix(".tiff"))
    if cand.exists():
        return cand
    cand = (msk_root / rel.with_suffix(".tif"))
    if cand.exists():
        return cand
    cand = (msk_root / rel)
    if cand.exists():
        return cand
    return None


def build_stem_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if is_image_file(p):
            s = normalize_stem(p.stem).lower()
            if s not in idx:
                idx[s] = p
    return idx


def pair_images_masks(root: Path) -> Tuple[List[Tuple[str, str]], dict]:
    pairs: List[Tuple[str, str]] = []
    bad_missing_mask: List[str] = []
    bad_read_error: List[str] = []
    bad_size_mismatch: List[Tuple[str, str, tuple, tuple]] = []
    size_hist: Dict[str, int] = {}


    candidate_pairs: List[Tuple[Path, Path]] = []
    for dirpath, dirnames, _ in os.walk(root):
        dirpath = Path(dirpath)

        imgs = [dirpath / d for d in dirnames if IMG_DIR_RE.search(d)]
        msks = [dirpath / d for d in dirnames if MSK_DIR_RE.search(d)]
        if not imgs or not msks:
            continue

        candidate_pairs.append((imgs[0], msks[0]))


    for img_root, msk_root in candidate_pairs:
        stem_index = build_stem_index(msk_root)

        for ip in img_root.rglob("*"):
            if not is_image_file(ip):
                continue


            rel = ip.relative_to(img_root)
            mp = try_mask_path(msk_root, rel)


            if mp is None:
                key = normalize_stem(ip.stem).lower()
                mp = stem_index.get(key, None)

            if mp is None or not mp.exists():
                bad_missing_mask.append(str(ip))
                continue

            try:
                im = io.imread(str(ip))
                mk = io.imread(str(mp))
                if im.ndim == 3 and im.shape[-1] == 1:
                    im = im[..., 0]
                if mk.ndim == 3 and mk.shape[-1] == 1:
                    mk = mk[..., 0]
            except Exception:
                bad_read_error.append(str(ip))
                continue

            if im.shape != mk.shape:
                bad_size_mismatch.append((str(ip), str(mp), im.shape, mk.shape))
                continue

            key_shape = str(im.shape)
            size_hist[key_shape] = size_hist.get(key_shape, 0) + 1
            pairs.append((str(ip), str(mp)))

    stats = {
        "root": str(root),
        "num_sibling_levels": len(candidate_pairs),
        "valid_pairs": len(pairs),
        "size_histogram": dict(sorted(size_hist.items(), key=lambda x: -x[1])),
        "bad_counts": {
            "missing_mask": len(bad_missing_mask),
            "read_error": len(bad_read_error),
            "size_mismatch": len(bad_size_mismatch),
        },
    }
    details = {
        "missing_mask_head": bad_missing_mask[:10],
        "read_error_head": bad_read_error[:10],
        "size_mismatch_head": bad_size_mismatch[:5],
    }
    return pairs, {"stats": stats, "details": details}


def save_overlays(pairs: List[Tuple[str, str]], outdir: Path, n: int = 16):
    if not HAS_PLT or not pairs:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    samp = random.sample(pairs, min(n, len(pairs)))
    for i, (ip, mp) in enumerate(samp):
        im = io.imread(ip); mk = io.imread(mp)
        if im.ndim == 3 and im.shape[-1] == 1: im = im[..., 0]
        if mk.ndim == 3 and mk.shape[-1] == 1: mk = mk[..., 0]
        imf = im.astype(np.float32)
        p1, p99 = np.percentile(imf, [1, 99])
        if p99 > p1:
            imf = np.clip((imf - p1) / (p99 - p1), 0, 1)
        rgb = np.dstack([imf, (mk > 0).astype(np.float32), np.zeros_like(imf)])
        plt.figure(figsize=(4, 4)); plt.axis("off")
        plt.imshow(rgb); plt.tight_layout(pad=0)
        plt.savefig(outdir / f"sample_{i:02d}.png", dpi=150)
        plt.close()


def main():
    print(f"[scan] ROOT = {ROOT}")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    pairs, info = pair_images_masks(ROOT)

    (QA_DIR / "pairs.txt").write_text(
        "\n".join([f"{a}\t{b}" for a, b in pairs]), encoding="utf-8"
    )

    with open(QA_DIR / "report-smart.json", "w", encoding="utf-8") as f:
        json.dump(info["stats"], f, indent=2)
    with open(QA_DIR / "report-smart-details.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info["stats"], indent=2))

    save_overlays(pairs, QA_DIR, n=16)
    if pairs:
        print(f"[scan] Wrote {len(pairs)} pairs to {QA_DIR/'pairs.txt'} and overlays to {QA_DIR}")
    else:
        print("[scan] No valid pairs found. Check details JSON and directory naming (images/ vs masks/).")


if __name__ == "__main__":
    main()
