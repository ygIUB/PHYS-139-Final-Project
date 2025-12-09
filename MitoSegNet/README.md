# MitoSegNet for EM Data

Complete implementation of MitoSegNet architecture trained on electron microscopy (EM) data for mitochondrial segmentation.

## Overview

This directory contains:
1. **MitoSegNet Architecture** - Modified U-Net with BatchNormalization (no Dropout)
2. **Training Pipeline** - Train on your EM data with comprehensive metrics
3. **Inference Scripts** - Both pretrained (fluorescent) and trained (EM) models
4. **Benchmarking** - Full evaluation with Dice, IoU, precision, recall, etc.

## Directory Structure

```
MitoSegNet/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── architecture.py              # MitoSegNet model definition
├── data_loader.py              # Data loading following MoDL_seg pattern
├── train.py                    # Training script with graphs
├── inference_pretrained.py     # Inference with original MitoSegNet (fluorescent)
├── inference_trained.py        # Inference with our trained model (EM)
├── benchmark.py                # Comprehensive benchmarking
├── venv/                       # Virtual environment
├── models/                     # Saved models (created during training)
├── outputs/                    # Inference outputs
└── benchmark_results/          # Evaluation results
```

## Setup

### 1. Activate Virtual Environment and Install Dependencies

```bash
cd MitoSegNet

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The data loader expects data in the same format as `MoDL_seg`:
- Training images: `../npydata/imgs_train.npy`
- Training masks: `../npydata/imgs_mask_train.npy`

If you haven't created these yet, run the data preparation from the main project.

## Usage

### Step 1: Train MitoSegNet on EM Data

Train the MitoSegNet architecture from scratch on your EM data:

```bash
python train.py
```

**What this does:**
- Loads data from `../npydata/`
- Builds MitoSegNet architecture (Modified U-Net)
- Trains for 30 epochs with early stopping
- Saves best model to `./models/mitosegnet_best.keras`
- Generates training curves (loss, accuracy)
- Creates training summary

**Outputs:**
- `models/mitosegnet_best.keras` - Best model (lowest val loss)
- `models/mitosegnet_final.keras` - Final model after all epochs
- `models/training_curves.png` - Combined accuracy/loss plot
- `models/training_accuracy.png` - Accuracy over epochs
- `models/training_loss.png` - Loss over epochs
- `models/training_summary.txt` - Text summary of training

**Configuration:**
- Image size: 512×512
- Max samples: 6000 (for GPU memory)
- Batch size: 2
- Epochs: 30 (with early stopping)
- Learning rate: 1e-4
- Validation split: 20%

### Step 2: Run Inference

#### Option A: Inference with Pretrained MitoSegNet (Fluorescent)

⚠️ **Note:** The pretrained model was trained on fluorescent microscopy and will perform poorly on EM data. This is useful for demonstrating domain mismatch.

```bash
python inference_pretrained.py
```

**Requirements:**
- Download pretrained MitoSegNet model (`.hdf5` file)
- Place at `./models/MitoSegNet_pretrained.hdf5`

#### Option B: Inference with Our Trained Model (EM)

Run inference using the model we trained on EM data:

```bash
python inference_trained.py
```

**What this does:**
- Loads trained model from `./models/mitosegnet_best.keras`
- Processes all images in `../test/images/`
- Saves binary masks to `./outputs/trained_predictions/binary/`
- Saves probability maps to `./outputs/trained_predictions/probability/`
- Creates sample visualizations

**Outputs:**
- `outputs/trained_predictions/binary/*.tif` - Binary segmentation masks
- `outputs/trained_predictions/probability/*.tif` - Probability maps
- `outputs/trained_predictions/sample_predictions.png` - Visualization grid

### Step 3: Benchmark Performance

Evaluate the model with comprehensive metrics:

```bash
python benchmark.py
```

**What this does:**
- Compares predictions with ground truth masks
- Calculates segmentation metrics:
  - Dice Coefficient
  - Intersection over Union (IoU)
  - Pixel Accuracy
  - Precision & Recall
- Extracts morphological features
- Generates plots and reports

**Outputs:**
- `benchmark_results/detailed_results.csv` - Per-image metrics
- `benchmark_results/summary_statistics.json` - Overall statistics
- `benchmark_results/metrics_distributions.png` - Metric histograms
- `benchmark_results/performance_bars.png` - Bar chart comparison
- `benchmark_results/benchmark_report.txt` - Text summary

**Configuration:**
Edit paths in `benchmark.py`:
```python
GT_DIR = "../test/masks"  # Ground truth masks
PRED_DIR = "./outputs/trained_predictions/binary"  # Predictions
```

## Architecture Details

### MitoSegNet vs Standard U-Net

**Key Difference:** BatchNormalization instead of Dropout

| Layer | Standard U-Net | MitoSegNet |
|-------|----------------|------------|
| Encoder blocks | Conv → Conv → MaxPool | Conv → Conv → **BatchNorm** → MaxPool |
| Bottleneck | Conv → Conv → Dropout | Conv → Conv → **BatchNorm** |
| Decoder blocks | Conv → Conv | Conv → Conv → **BatchNorm** |

**Why this matters:**
- BatchNormalization improves training stability
- Better validation loss and dice coefficient (per original paper)
- More suitable for small batch sizes (batch=2)

### Model Summary

```
Input: (512, 512, 1)
├─ Encoder
│  ├─ Block 1: 64 filters → BatchNorm → Pool
│  ├─ Block 2: 128 filters → BatchNorm → Pool
│  ├─ Block 3: 256 filters → BatchNorm → Pool
│  └─ Block 4: 512 filters → BatchNorm → Pool
├─ Bottleneck: 1024 filters → BatchNorm
└─ Decoder
   ├─ Block 6: UpConv → Concat → 512 filters → BatchNorm
   ├─ Block 7: UpConv → Concat → 256 filters → BatchNorm
   ├─ Block 8: UpConv → Concat → 128 filters → BatchNorm
   └─ Block 9: UpConv → Concat → 64 filters → BatchNorm
Output: (512, 512, 1) with sigmoid activation
```

Total parameters: ~31 million

## Metrics Explained

### Segmentation Metrics

- **Dice Coefficient**: Harmonic mean of precision and recall (0-1, higher is better)
  - Formula: `2×|Pred ∩ GT| / (|Pred| + |GT|)`
  - Interpretation: 0.8+ is excellent, 0.6-0.8 is good

- **IoU (Intersection over Union)**: Overlap ratio (0-1, higher is better)
  - Formula: `|Pred ∩ GT| / |Pred ∪ GT|`
  - Related to Dice: `Dice = 2×IoU / (1 + IoU)`

- **Pixel Accuracy**: Percentage of correctly classified pixels (0-1)
  - Can be misleading with class imbalance

- **Precision**: True positives / (True positives + False positives)
  - "Of pixels predicted as mitochondria, what fraction are correct?"

- **Recall**: True positives / (True positives + False negatives)
  - "Of actual mitochondria pixels, what fraction did we detect?"

### Morphological Features

- **Count**: Number of individual mitochondria detected
- **Mean Area**: Average size of detected mitochondria
- **Total Area**: Total mitochondrial content
- **Eccentricity**: How elongated mitochondria are (0=circle, 1=line)
- **Solidity**: How "solid" vs "fragmented" (convex hull ratio)

## Comparison with Other Models

### vs Baseline U-Net

- Baseline U-Net: Uses Dropout (0.5) in bottleneck layers
- MitoSegNet: Uses BatchNormalization throughout
- **Expected:** Similar or slightly better performance with MitoSegNet

### vs OurResUNet

- OurResUNet: Residual connections + dilated convolutions + LayerNorm
- MitoSegNet: Standard U-Net + BatchNorm
- **Expected:** OurResUNet should outperform (more sophisticated architecture)

### vs Pretrained MitoSegNet (Fluorescent)

- Pretrained: Trained on fluorescent C. elegans images
- Our trained: Trained on EM data
- **Expected:** Pretrained will fail (domain mismatch), our trained will succeed

## Common Issues & Solutions

### 1. Out of Memory Error

**Problem:** GPU runs out of memory during training

**Solutions:**
- Reduce `MAX_SAMPLES` in `train.py` (default: 6000)
- Reduce `BATCH_SIZE` (default: 2)
- Use CPU instead: `tf.config.set_visible_devices([], 'GPU')`

### 2. Data Not Found

**Problem:** FileNotFoundError for .npy files

**Solution:** Make sure data is prepared:
```bash
cd ../MoDL_seg
python data_load.py  # Creates ../npydata/*.npy files
```

### 3. Model File Not Found

**Problem:** Can't find `mitosegnet_best.keras`

**Solution:** Train the model first:
```bash
python train.py
```

### 4. Slow Training

**Problem:** Training takes too long

**Solutions:**
- Ensure GPU is being used: Check for "GPU available" in output
- Reduce number of epochs
- Reduce max samples

## Workflow Summary

**Full Pipeline:**
```bash
# 1. Setup
source venv/bin/activate
pip install -r requirements.txt

# 2. Train model on EM data
python train.py

# 3. Run inference
python inference_trained.py

# 4. Evaluate performance
python benchmark.py

# 5. (Optional) Try pretrained model to show domain mismatch
python inference_pretrained.py  # Requires pretrained .hdf5 file
```

## References

- **MitoSegNet Paper:** Viana, M. P., et al. (2020). "MitoSegNet: Easy-to-use Deep Learning Segmentation for Analyzing Mitochondrial Morphology." *iScience*, 23(10), 101601.
- **U-Net Paper:** Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."

## Notes

- This implementation adapts MitoSegNet architecture for EM data
- Original MitoSegNet was trained on 12 fluorescent images
- Our version trains from scratch on thousands of EM images
- Architecture is proven to work well for mitochondrial segmentation
- BatchNormalization key improvement over standard U-Net