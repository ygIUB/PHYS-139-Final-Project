import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "cem_mitolab"
OUTDIR = ROOT / "npydata"
OUTDIR.mkdir(exist_ok=True, parents=True)

EXTS = {".tif", ".tiff", ".png"}


def read_gray_hw1(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    a = np.array(im, dtype=np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    elif a.ndim == 3:
        a = a[..., 0:1]
    return a


def fit512(a: np.ndarray) -> np.ndarray:
    H, W, C = a.shape
    ph, pw = max(0, 512 - H), max(0, 512 - W)
    if ph > 0 or pw > 0:
        a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        H, W, C = a.shape
    y0 = (H - 512) // 2
    x0 = (W - 512) // 2
    return a[y0:y0 + 512, x0:x0 + 512, :]


def collect_pairs(data_root: Path):
    if not data_root.exists():
        raise SystemExit(f"[error] 数据目录不存在: {data_root}")

    Xs, Ys = [], []
    total, bad = 0, 0

    # 遍历每个 case 子目录
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue

        img_dir = case_dir / "images"
        msk_dir = case_dir / "masks"
        if not (img_dir.exists() and msk_dir.exists()):
            # 有的子目录可能不是数据，直接跳过
            continue

        # 以 mask 目录建立索引：stem -> Path
        mindex = {
            p.stem: p for p in msk_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        }

        # 在 images 目录里找与之同名的图像
        img_paths = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS and p.stem in mindex
        ]
        img_paths = sorted(img_paths, key=lambda p: (case_dir.name, p.stem))

        print(f"[case] {case_dir.name}: {len(img_paths)} pairs")

        for ip in img_paths:
            mp = mindex[ip.stem]
            total += 1
            try:
                im = fit512(read_gray_hw1(ip))
                mk = fit512(read_gray_hw1(mp))
                # 非 0 当作前景，二值化到 {0,255}
                mk = (mk > 0).astype(np.uint8) * 255
                Xs.append(im)
                Ys.append(mk)
            except Exception as e:
                bad += 1
                if bad <= 10:
                    print(f"[skip] {case_dir.name}/{ip.name}: {e}", file=sys.stderr)

    if not Xs:
        raise SystemExit("[error] 没有找到任何成对的 (image, mask)，请检查目录结构和文件名。")

    X = np.stack(Xs, axis=0).astype(np.uint8)
    Y = np.stack(Ys, axis=0).astype(np.uint8)
    print(f"[pack] total pairs: {total}, used: {len(Xs)}, skipped: {bad}")
    return X, Y


def main():
    X, Y = collect_pairs(DATA_ROOT)
    np.save(OUTDIR / "imgs_train.npy", X)
    np.save(OUTDIR / "imgs_mask_train.npy", Y)
    print("[pack] saved:")
    print("       ", OUTDIR / "imgs_train.npy", X.shape, X.dtype)
    print("       ", OUTDIR / "imgs_mask_train.npy", Y.shape, Y.dtype)


if __name__ == "__main__":
    main()
