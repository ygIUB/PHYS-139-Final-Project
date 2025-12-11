# split_npy_80_10_10.py
# 读取 npydata/imgs_train.npy & imgs_mask_train.npy
# 按 0.8 / 0.1 / 0.1 随机划分 train / val / test

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NPYDIR = ROOT / "npydata"

X_path = NPYDIR / "imgs_train.npy"
Y_path = NPYDIR / "imgs_mask_train.npy"

print("[load]", X_path)
print("[load]", Y_path)
X = np.load(X_path)
Y = np.load(Y_path)

assert X.shape[0] == Y.shape[0], "X 和 Y 的样本数量不一致！"
N = X.shape[0]
print(f"[info] 总样本数 N = {N}")

# 为了可复现，可以固定随机种子
rng = np.random.default_rng(seed=42)
idx = rng.permutation(N)

n_train = int(N * 0.8)
n_val   = int(N * 0.1)
n_test  = N - n_train - n_val   # 剩下的都给 test，保证总数对得上

i_train = idx[:n_train]
i_val   = idx[n_train:n_train + n_val]
i_test  = idx[n_train + n_val:]

splits = {
    "train": i_train,
    "val":   i_val,
    "test":  i_test,
}

for name, inds in splits.items():
    X_split = X[inds]
    Y_split = Y[inds]
    np.save(NPYDIR / f"imgs_{name}.npy", X_split)
    np.save(NPYDIR / f"masks_{name}.npy", Y_split)
    print(f"[save] imgs_{name}.npy 形状: {X_split.shape}")
    print(f"[save] masks_{name}.npy 形状: {Y_split.shape}")

print("[done] 划分完成：train/val/test =",
      splits["train"].size, splits["val"].size, splits["test"].size)
