# MoDL-style Mitochondria Segmentation (CEM-MitoLab, Windows + DirectML)

**Current pipeline:**

- OS: Windows 10/11
- Python: 3.10
- Environment: Conda
- Deep learning: TensorFlow 2.10 + `tensorflow-directml-plugin` (DirectML GPU backend)
- Data: **CEM-MitoLab** EM mitochondria dataset (`cem_mitolab/**/images` + `masks`)
- Task: 2D EM **mitochondria segmentation** (no supervised function prediction yet)

---

## 1. Repository layout

At the moment the project is organized roughly as:

```text
D:\MoDL-main
├── MoDL_seg/              # U-Net / U-RNet+ model + training script (train.py, etc.)
├── cem_mitolab/           # CEM-MitoLab EM data, many subfolders, each with images/ & masks/
│   ├── case_xxxx/
│   │   ├── images/*.tiff
│   │   └── masks/*.tiff
│   └── ...
├── npydata/               # NPY arrays generated from cem_mitolab
│   ├── imgs_train.npy
│   └── imgs_mask_train.npy
├── model/                 # Training outputs (weights + curves)
│   ├── U-RNet+_gpu_10ep.keras
│   ├── Accuracy.png
│   ├── Loss.png
│   └── unet.txt
├── pack_512_only.py       # Build (N, 512, 512, 1) NPY from cem_mitolab images/masks
├── split_npy_80_10_10.py  # (Optional) Split NPY into train/val/test (0.8 / 0.1 / 0.1)
└── README.md              # This file
