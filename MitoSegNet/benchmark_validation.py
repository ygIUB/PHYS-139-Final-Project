"""
Benchmark MitoSegNet on the validation set (seed=123 split)
Ensures we benchmark on data NOT used for training
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from skimage import measure
import tensorflow as tf
from tensorflow.keras.models import load_model
from architecture import dice_coefficient

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check GPU availability
print("=" * 60)
print("CHECKING GPU AVAILABILITY")
print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    print("TensorFlow will use GPU for inference")
else:
    print("⚠ No GPU found - will use CPU (slower)")
print("=" * 60 + "\n")


class ValidationBenchmark:
    """Benchmark model on validation set from training split"""

    def __init__(self, model_path, npy_path="../npydata", output_dir="./benchmark_results"):
        """
        Initialize benchmark

        Args:
            model_path: Path to trained model (.keras file)
            npy_path: Path to .npy data files
            output_dir: Output directory for results
        """
        self.model_path = model_path
        self.npy_path = npy_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.model = None
        self.results = []

    def load_model(self):
        """Load trained model"""
        print("=" * 60)
        print("LOADING MODEL")
        print("=" * 60)

        print(f"Loading model from: {self.model_path}")
        self.model = load_model(
            self.model_path,
            custom_objects={'dice_coefficient': dice_coefficient}
        )
        print(f"Model loaded: {self.model.name}")
        print(f"Parameters: {self.model.count_params():,}")

    def load_validation_data(self, max_samples=6000, val_ratio=0.2):
        """
        Load validation data using the SAME split as training

        Args:
            max_samples: Max samples (should match training)
            val_ratio: Validation ratio (should match training)

        Returns:
            tuple: (X_val, Y_val) - validation images and masks
        """
        print("\n" + "=" * 60)
        print("LOADING VALIDATION DATA")
        print("=" * 60)

        # Load full dataset with memory mapping
        imgs_mmap = np.load(self.npy_path + "/imgs_train.npy", mmap_mode='r')
        masks_mmap = np.load(self.npy_path + "/imgs_mask_train.npy", mmap_mode='r')

        N = imgs_mmap.shape[0]
        print(f"Total samples in dataset: {N}")

        # STEP 1: Apply same subsampling as training (seed=42)
        if N > max_samples:
            print(f"Subsampling to {max_samples} (matching training)...")
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(N, size=max_samples, replace=False)
            idx = np.sort(idx)

            imgs = imgs_mmap[idx].copy()
            masks = masks_mmap[idx].copy()
        else:
            imgs = imgs_mmap[:].copy()
            masks = masks_mmap[:].copy()

        print(f"Loaded {imgs.shape[0]} samples")

        # STEP 2: Preprocess (same as training)
        print("Preprocessing...")
        imgs = imgs.astype('float16')
        masks = masks.astype('float16')

        imgs /= 255.0
        mean = imgs.mean(axis=0, dtype='float32')
        imgs = imgs - mean.astype('float16')

        masks /= 255.0
        masks[masks > 0.5] = 1.0
        masks[masks <= 0.5] = 0.0

        # STEP 3: Apply SAME split as training (seed=123)
        N = imgs.shape[0]
        val_size = int(N * val_ratio)

        # CRITICAL: Use seed=123 to match training split
        rng = np.random.default_rng(seed=123)
        indices = rng.permutation(N)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_val = imgs[val_idx]
        Y_val = masks[val_idx]

        print(f"Train samples: {len(train_idx)} (NOT used for benchmarking)")
        print(f"Val samples: {len(val_idx)} (USED for benchmarking)")
        print(f"Validation shape: {X_val.shape}")

        return X_val, Y_val

    def run_inference(self, X_val):
        """
        Run inference on validation data

        Args:
            X_val: Validation images

        Returns:
            Predictions (binary masks)
        """
        print("\n" + "=" * 60)
        print("RUNNING INFERENCE")
        print("=" * 60)

        print(f"Predicting on {len(X_val)} validation images...")
        predictions = self.model.predict(X_val, batch_size=2, verbose=1)

        # Binarize predictions (threshold at 0.5)
        predictions_binary = (predictions > 0.5).astype(np.uint8)

        print(f"Predictions shape: {predictions_binary.shape}")

        return predictions_binary

    def calculate_metrics(self, pred, gt):
        """
        Calculate segmentation metrics for a single image

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask

        Returns:
            dict: Metrics
        """
        pred_flat = pred.astype(bool).flatten()
        gt_flat = gt.astype(bool).flatten()

        # Dice coefficient
        intersection = np.sum(pred_flat & gt_flat)
        dice = (2.0 * intersection + 1e-6) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-6)

        # IoU
        union = np.sum(pred_flat | gt_flat)
        iou = (intersection + 1e-6) / (union + 1e-6)

        # Pixel accuracy
        correct = np.sum(pred_flat == gt_flat)
        pixel_acc = correct / len(pred_flat)

        # Precision and Recall
        tp = intersection
        fp = np.sum(pred_flat & ~gt_flat)
        fn = np.sum(~pred_flat & gt_flat)

        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)

        return {
            'dice': float(dice),
            'iou': float(iou),
            'pixel_accuracy': float(pixel_acc),
            'precision': float(precision),
            'recall': float(recall)
        }

    def extract_morphological_features(self, mask):
        """
        Extract morphological features from binary mask

        Args:
            mask: Binary mask

        Returns:
            dict: Morphological features
        """
        # Remove channel dimension if present
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Label connected components
        labeled = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled)

        if len(regions) == 0:
            return {
                'count': 0,
                'total_area': 0,
                'mean_area': 0,
                'mean_perimeter': 0,
                'mean_eccentricity': 0,
                'mean_solidity': 0
            }

        areas = [r.area for r in regions]
        perimeters = [r.perimeter for r in regions]
        eccentricities = [r.eccentricity for r in regions]
        solidities = [r.solidity for r in regions]

        return {
            'count': len(regions),
            'total_area': sum(areas),
            'mean_area': np.mean(areas),
            'mean_perimeter': np.mean(perimeters),
            'mean_eccentricity': np.mean(eccentricities),
            'mean_solidity': np.mean(solidities)
        }

    def evaluate(self, predictions, Y_val):
        """
        Evaluate all predictions against ground truth

        Args:
            predictions: Predicted masks
            Y_val: Ground truth masks
        """
        print("\n" + "=" * 60)
        print("EVALUATING PREDICTIONS")
        print("=" * 60)

        self.results = []

        for i in tqdm(range(len(predictions)), desc="Evaluating"):
            pred = predictions[i]
            gt = Y_val[i]

            # Calculate metrics
            metrics = self.calculate_metrics(pred, gt)

            # Extract morphological features
            pred_morph = self.extract_morphological_features(pred)
            gt_morph = self.extract_morphological_features(gt)

            # Combine results
            result = {
                'sample_idx': i,
                **metrics
            }

            # Add morphological features
            for key, value in pred_morph.items():
                result[f'pred_{key}'] = value
            for key, value in gt_morph.items():
                result[f'gt_{key}'] = value

            self.results.append(result)

        print(f"\nEvaluated {len(self.results)} validation samples")

    def calculate_statistics(self):
        """Calculate summary statistics"""
        df = pd.DataFrame(self.results)

        stats = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'dice_mean': float(df['dice'].mean()),
            'dice_std': float(df['dice'].std()),
            'dice_median': float(df['dice'].median()),
            'dice_min': float(df['dice'].min()),
            'dice_max': float(df['dice'].max()),
            'iou_mean': float(df['iou'].mean()),
            'iou_std': float(df['iou'].std()),
            'iou_median': float(df['iou'].median()),
            'pixel_accuracy_mean': float(df['pixel_accuracy'].mean()),
            'pixel_accuracy_std': float(df['pixel_accuracy'].std()),
            'precision_mean': float(df['precision'].mean()),
            'precision_std': float(df['precision'].std()),
            'recall_mean': float(df['recall'].mean()),
            'recall_std': float(df['recall'].std()),
        }

        return stats

    def plot_results(self):
        """Generate visualization plots"""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        df = pd.DataFrame(self.results)

        # 1. Metrics distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall']

        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            ax.hist(df[metric], bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(df[metric].mean(), color='r', linestyle='--',
                      label=f'Mean: {df[metric].mean():.3f}')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        axes[1, 2].remove()

        plt.tight_layout()
        plot_path = self.output_dir / 'metrics_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

        # 2. Summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        means = [df[m].mean() for m in metrics]
        stds = [df[m].std() for m in metrics]

        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')

        # Color bars
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=15, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Validation Set Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.output_dir / 'performance_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

    def save_results(self):
        """Save all results"""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        # Save detailed results
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'validation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Calculate and save statistics
        stats = self.calculate_statistics()
        stats_path = self.output_dir / 'validation_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {stats_path}")

        # Print statistics
        print("\n" + "=" * 60)
        print("VALIDATION SET PERFORMANCE")
        print("=" * 60)
        print(f"Samples: {stats['n_samples']}")
        print(f"\nDice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}")
        print(f"  Median: {stats['dice_median']:.4f}")
        print(f"  Range: [{stats['dice_min']:.4f}, {stats['dice_max']:.4f}]")
        print(f"\nIoU:              {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
        print(f"Pixel Accuracy:   {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}")
        print(f"Precision:        {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
        print(f"Recall:           {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")

        # Save text report
        report_path = self.output_dir / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MITOSEGNET VALIDATION SET BENCHMARK\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Validation samples: {stats['n_samples']}\n\n")

            f.write("SEGMENTATION METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Dice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}\n")
            f.write(f"  Median: {stats['dice_median']:.4f}\n")
            f.write(f"  Range: [{stats['dice_min']:.4f}, {stats['dice_max']:.4f}]\n\n")
            f.write(f"IoU:              {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}\n")
            f.write(f"Pixel Accuracy:   {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}\n")
            f.write(f"Precision:        {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}\n")
            f.write(f"Recall:           {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("NOTE: Benchmarked on validation set (20%) from seed=123 split\n")
            f.write("This data was NOT used during training.\n")
            f.write("=" * 60 + "\n")

        print(f"Saved: {report_path}")

        return stats


def main():
    """Main benchmarking function"""
    print("\n" + "=" * 60)
    print("MITOSEGNET VALIDATION BENCHMARK")
    print("=" * 60)
    print("\nThis script benchmarks on the VALIDATION SET ONLY")
    print("Using the same seed=123 split from training")
    print("Ensures we test on data NOT used for training\n")

    # Configuration
    MODEL_PATH = "./models/mitosegnet_best.keras"
    NPY_PATH = "../npydata"
    OUTPUT_DIR = "./benchmark_results"
    MAX_SAMPLES = 6000  # Must match training
    VAL_RATIO = 0.2     # Must match training

    # Initialize benchmark
    benchmark = ValidationBenchmark(
        model_path=MODEL_PATH,
        npy_path=NPY_PATH,
        output_dir=OUTPUT_DIR
    )

    # Load model
    benchmark.load_model()

    # Load validation data (same split as training)
    X_val, Y_val = benchmark.load_validation_data(
        max_samples=MAX_SAMPLES,
        val_ratio=VAL_RATIO
    )

    # Run inference
    predictions = benchmark.run_inference(X_val)

    # Evaluate
    benchmark.evaluate(predictions, Y_val)

    # Plot results
    benchmark.plot_results()

    # Save results
    stats = benchmark.save_results()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"Main metric - Dice: {stats['dice_mean']:.4f}")


if __name__ == "__main__":
    main()
