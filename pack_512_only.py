import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "cem_mitolab"
OUTDIR = ROOT / "npydata"
OUTDIR.mkdir(exist_ok=True, parents=True)

EXTS = {".tif", ".tiff", ".png"}


def read_gray(path: Path) -> np.ndarray:
    """读图 -> 灰度 -> (H, W) uint8"""
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return np.array(im, dtype=np.uint8)


def collect_pairs_512(data_root: Path):
    if not data_root.exists():
        raise SystemExit(f"[error] 数据目录不存在: {data_root}")

    Xs, Ys = [], []
    total_pairs = 0
    used_512 = 0
    skipped_size = 0
    skipped_err = 0

    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue

        img_dir = case_dir / "images"
        msk_dir = case_dir / "masks"
        if not (img_dir.exists() and msk_dir.exists()):
            continue

        mindex = {
            p.stem: p for p in msk_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        }

        img_paths = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS and p.stem in mindex
        ]
        img_paths = sorted(img_paths, key=lambda p: (case_dir.name, p.stem))

        print(f"[case] {case_dir.name}: {len(img_paths)} raw pairs")
        for ip in img_paths:
            mp = mindex[ip.stem]
            total_pairs += 1
            try:
                im = read_gray(ip)
                mk = read_gray(mp)

                if im.shape != (512, 512) or mk.shape != (512, 512):
                    skipped_size += 1
                    continue

                im = im[..., None]
                mk = (mk > 0).astype(np.uint8)[..., None] * 255

                Xs.append(im)
                Ys.append(mk)
                used_512 += 1
            except Exception as e:
                skipped_err += 1
                if skipped_err <= 10:
                    print(f"[skip-err] {case_dir.name}/{ip.name}: {e}", file=sys.stderr)

    if not Xs:
        raise SystemExit("[error] 没有任何 512x512 的成对样本，请检查数据。")

    X = np.stack(Xs, axis=0).astype(np.uint8)
    Y = np.stack(Ys, axis=0).astype(np.uint8)

    print(f"[summary] total raw pairs   : {total_pairs}")
    print(f"[summary] used 512x512 pairs: {used_512}")
    print(f"[summary] skipped by size   : {skipped_size}")
    print(f"[summary] skipped by error  : {skipped_err}")

    return X, Y


def main():
    X, Y = collect_pairs_512(DATA_ROOT)
    np.save(OUTDIR / "imgs_train.npy", X)
    np.save(OUTDIR / "imgs_mask_train.npy", Y)
    print("[save] ", OUTDIR / "imgs_train.npy", X.shape, X.dtype)
    print("[save] ", OUTDIR / "imgs_mask_train.npy", Y.shape, Y.dtype)


if __name__ == "__main__":
    main()
