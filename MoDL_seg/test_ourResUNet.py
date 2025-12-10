"""
Benchmark OurResUnet/Baseline U-Net on the validation set (seed=123 split)
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
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from tensorflow.keras.models import load_model

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


class InstanceSegmentationMetrics:
    """
    Calculate instance-level segmentation metrics for comparison with published results.

    NOTE: These metrics are from the CEM-MitoLab paper (Conrad & Narayan, 2023):
    "CEM-MitoLab: A domain-adaptive self-supervised method for mitochondria segmentation"

    Metrics match those reported in their Table 2:
    - F1@50, F1@75: F1 scores at IoU thresholds 0.50 and 0.75
    - AP@50, AP@75: Average Precision at IoU thresholds 0.50 and 0.75
    - PQ: Panoptic Quality for instance segmentation

    These are instance-based metrics (object-level), not pixel-based metrics.
    """

    @staticmethod
    def label_instances(binary_mask):
        """
        Label connected components in binary mask

        Args:
            binary_mask: Binary segmentation mask

        Returns:
            labeled_mask: Instance-labeled mask
            num_instances: Number of instances found
        """
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.squeeze()

        labeled_mask = measure.label(binary_mask, connectivity=2)
        num_instances = labeled_mask.max()

        return labeled_mask, num_instances

    @staticmethod
    def compute_iou_matrix(pred_labeled, gt_labeled):
        """
        Compute IoU matrix between all predicted and ground truth instances

        Args:
            pred_labeled: Labeled prediction mask
            gt_labeled: Labeled ground truth mask

        Returns:
            iou_matrix: Matrix of IoU scores [n_pred, n_gt]
        """
        pred_instances = np.unique(pred_labeled)[1:]  # Exclude background (0)
        gt_instances = np.unique(gt_labeled)[1:]

        n_pred = len(pred_instances)
        n_gt = len(gt_instances)

        if n_pred == 0 or n_gt == 0:
            return np.zeros((n_pred, n_gt))

        iou_matrix = np.zeros((n_pred, n_gt))

        for i, pred_id in enumerate(pred_instances):
            pred_mask = (pred_labeled == pred_id)
            for j, gt_id in enumerate(gt_instances):
                gt_mask = (gt_labeled == gt_id)

                intersection = np.sum(pred_mask & gt_mask)
                union = np.sum(pred_mask | gt_mask)

                if union > 0:
                    iou_matrix[i, j] = intersection / union

        return iou_matrix

    @staticmethod
    def match_instances(iou_matrix, iou_threshold=0.5):
        """
        Match predicted instances to ground truth using Hungarian algorithm

        Args:
            iou_matrix: IoU matrix [n_pred, n_gt]
            iou_threshold: Minimum IoU for a valid match

        Returns:
            matches: List of (pred_idx, gt_idx, iou) tuples
            unmatched_pred: List of unmatched prediction indices
            unmatched_gt: List of unmatched ground truth indices
        """
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        # Use Hungarian algorithm for optimal matching
        # We want to maximize IoU, so use negative for minimization
        pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)

        matches = []
        matched_pred = set()
        matched_gt = set()

        for p_idx, g_idx in zip(pred_idx, gt_idx):
            iou = iou_matrix[p_idx, g_idx]
            if iou >= iou_threshold:
                matches.append((p_idx, g_idx, iou))
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)

        # Find unmatched instances
        unmatched_pred = [i for i in range(iou_matrix.shape[0]) if i not in matched_pred]
        unmatched_gt = [i for i in range(iou_matrix.shape[1]) if i not in matched_gt]

        return matches, unmatched_pred, unmatched_gt

    @staticmethod
    def compute_f1_at_threshold(pred_mask, gt_mask, iou_threshold=0.5):
        """
        Compute F1 score at a specific IoU threshold (paper metric)

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            iou_threshold: IoU threshold for matching

        Returns:
            f1: F1 score
            precision: Precision
            recall: Recall
        """
        # Label instances
        pred_labeled, n_pred = InstanceSegmentationMetrics.label_instances(pred_mask)
        gt_labeled, n_gt = InstanceSegmentationMetrics.label_instances(gt_mask)

        if n_pred == 0 and n_gt == 0:
            return 1.0, 1.0, 1.0  # Perfect match when both are empty

        if n_pred == 0 or n_gt == 0:
            return 0.0, 0.0, 0.0  # No match possible

        # Compute IoU matrix and match
        iou_matrix = InstanceSegmentationMetrics.compute_iou_matrix(pred_labeled, gt_labeled)
        matches, unmatched_pred, unmatched_gt = InstanceSegmentationMetrics.match_instances(
            iou_matrix, iou_threshold
        )

        # Calculate metrics
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall

    @staticmethod
    def compute_average_precision_at_threshold(pred_mask, gt_mask, iou_threshold=0.5):
        """
        Compute Average Precision at a specific IoU threshold (paper metric)

        For single-image AP, this is equivalent to precision at the given threshold
        when we have confidence scores. Without confidence scores, we use F1 as proxy.

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            iou_threshold: IoU threshold for matching

        Returns:
            ap: Average Precision score
        """
        # Label instances
        pred_labeled, n_pred = InstanceSegmentationMetrics.label_instances(pred_mask)
        gt_labeled, n_gt = InstanceSegmentationMetrics.label_instances(gt_mask)

        if n_pred == 0 and n_gt == 0:
            return 1.0  # Perfect when both empty

        if n_gt == 0:
            return 0.0  # False positives with no ground truth

        if n_pred == 0:
            return 0.0  # No detections

        # Compute IoU matrix and match
        iou_matrix = InstanceSegmentationMetrics.compute_iou_matrix(pred_labeled, gt_labeled)
        matches, unmatched_pred, unmatched_gt = InstanceSegmentationMetrics.match_instances(
            iou_matrix, iou_threshold
        )

        # Calculate precision (which equals AP for binary predictions)
        tp = len(matches)
        fp = len(unmatched_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return precision

    @staticmethod
    def compute_panoptic_quality(pred_mask, gt_mask, iou_threshold=0.5):
        """
        Compute Panoptic Quality (PQ) - paper metric

        PQ = (sum of IoU for matched instances) / (TP + 0.5*FP + 0.5*FN)

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            iou_threshold: IoU threshold for matching

        Returns:
            pq: Panoptic Quality score
            sq: Segmentation Quality (mean IoU of matched instances)
            rq: Recognition Quality (F1 score)
        """
        # Label instances
        pred_labeled, n_pred = InstanceSegmentationMetrics.label_instances(pred_mask)
        gt_labeled, n_gt = InstanceSegmentationMetrics.label_instances(gt_mask)

        if n_pred == 0 and n_gt == 0:
            return 1.0, 1.0, 1.0  # Perfect when both empty

        if n_pred == 0 or n_gt == 0:
            return 0.0, 0.0, 0.0  # No quality when one is empty

        # Compute IoU matrix and match
        iou_matrix = InstanceSegmentationMetrics.compute_iou_matrix(pred_labeled, gt_labeled)
        matches, unmatched_pred, unmatched_gt = InstanceSegmentationMetrics.match_instances(
            iou_matrix, iou_threshold
        )

        # Calculate components
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        # Segmentation Quality (SQ): mean IoU of matched pairs
        if tp > 0:
            sum_iou = sum(iou for _, _, iou in matches)
            sq = sum_iou / tp
        else:
            sq = 0.0

        # Recognition Quality (RQ): F1 score
        if (tp + 0.5 * fp + 0.5 * fn) > 0:
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        else:
            rq = 0.0

        # Panoptic Quality
        pq = sq * rq

        return pq, sq, rq


class ValidationBenchmark:
    """Benchmark model on validation set from training split"""

    def __init__(self, model_path, model_name="OurResUNet", npy_path="../npydata", output_dir="./benchmark_results"):
        """
        Initialize benchmark

        Args:
            model_path: Path to trained model (.keras or .hdf5 file)
            model_name: Name of the model (for display purposes)
            npy_path: Path to .npy data files
            output_dir: Output directory for results
        """
        self.model_path = model_path
        self.model_name = model_name
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
        self.model = load_model(self.model_path)
        print(f"Model loaded: {self.model.name}")
        print(f"Parameters: {self.model.count_params():,}")

    def load_validation_data(self, max_samples=6000, val_ratio=0.2):
        """
        Load validation data using the SAME split as training
        (Must match the split used in train.py)

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
        Includes both standard metrics and CEM-MitoLab paper metrics

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask

        Returns:
            dict: Metrics
        """
        pred_flat = pred.astype(bool).flatten()
        gt_flat = gt.astype(bool).flatten()

        # Standard pixel-based metrics
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

        # Instance-based metrics (from CEM-MitoLab paper)
        # F1 at IoU thresholds
        f1_50, _, _ = InstanceSegmentationMetrics.compute_f1_at_threshold(pred, gt, iou_threshold=0.5)
        f1_75, _, _ = InstanceSegmentationMetrics.compute_f1_at_threshold(pred, gt, iou_threshold=0.75)

        # Average Precision at IoU thresholds
        ap_50 = InstanceSegmentationMetrics.compute_average_precision_at_threshold(pred, gt, iou_threshold=0.5)
        ap_75 = InstanceSegmentationMetrics.compute_average_precision_at_threshold(pred, gt, iou_threshold=0.75)

        # Panoptic Quality
        pq, sq, rq = InstanceSegmentationMetrics.compute_panoptic_quality(pred, gt, iou_threshold=0.5)

        return {
            # Standard metrics
            'dice': float(dice),
            'iou': float(iou),
            'pixel_accuracy': float(pixel_acc),
            'precision': float(precision),
            'recall': float(recall),
            # Paper metrics (CEM-MitoLab Table 2)
            'f1_50': float(f1_50),
            'f1_75': float(f1_75),
            'ap_50': float(ap_50),
            'ap_75': float(ap_75),
            'pq': float(pq),
            'sq': float(sq),
            'rq': float(rq)
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

            # Store results
            result = {
                'sample_idx': i,
                **metrics
            }

            self.results.append(result)

        print(f"\nEvaluated {len(self.results)} validation samples")

    def calculate_statistics(self):
        """Calculate summary statistics including paper metrics"""
        df = pd.DataFrame(self.results)

        stats = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'n_samples': len(df),
            # Standard metrics
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
            # Paper metrics (CEM-MitoLab Table 2)
            'f1_50_mean': float(df['f1_50'].mean()),
            'f1_50_std': float(df['f1_50'].std()),
            'f1_75_mean': float(df['f1_75'].mean()),
            'f1_75_std': float(df['f1_75'].std()),
            'ap_50_mean': float(df['ap_50'].mean()),
            'ap_50_std': float(df['ap_50'].std()),
            'ap_75_mean': float(df['ap_75'].mean()),
            'ap_75_std': float(df['ap_75'].std()),
            'pq_mean': float(df['pq'].mean()),
            'pq_std': float(df['pq'].std()),
            'sq_mean': float(df['sq'].mean()),
            'sq_std': float(df['sq'].std()),
            'rq_mean': float(df['rq'].mean()),
            'rq_std': float(df['rq'].std()),
        }

        return stats

    def plot_results(self):
        """Generate visualization plots"""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)

        df = pd.DataFrame(self.results)

        # 1. Standard Metrics distribution
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

        # 2. Paper Metrics distribution (CEM-MitoLab Table 2 metrics)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - CEM-MitoLab Paper Metrics (Table 2)',
                    fontsize=16, fontweight='bold', y=0.995)

        paper_metrics = ['f1_50', 'f1_75', 'ap_50', 'ap_75', 'pq']

        for i, metric in enumerate(paper_metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            ax.hist(df[metric], bins=30, edgecolor='black', alpha=0.7, color='#e67e22')
            ax.axvline(df[metric].mean(), color='r', linestyle='--',
                      label=f'Mean: {df[metric].mean():.3f}')
            ax.set_xlabel(metric.replace('_', '@').upper() if '@' in metric.replace('_', '@') else metric.upper())
            ax.set_ylabel('Count')
            ax.set_title(f'{metric.replace("_", "@").upper()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        axes[1, 2].remove()

        plt.tight_layout()
        plot_path = self.output_dir / 'paper_metrics_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

        # 3. Summary bar chart - Standard Metrics
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
        ax.set_title(f'{self.model_name} - Standard Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.output_dir / 'performance_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

        # 4. Summary bar chart - Paper Metrics (CEM-MitoLab Table 2)
        fig, ax = plt.subplots(figsize=(12, 6))

        paper_means = [df[m].mean() for m in paper_metrics]
        paper_stds = [df[m].std() for m in paper_metrics]

        x_pos = np.arange(len(paper_metrics))
        bars = ax.bar(x_pos, paper_means, yerr=paper_stds, capsize=5, alpha=0.7, edgecolor='black', color='#e67e22')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', '@').upper() if '@' in m.replace('_', '@') else m.upper()
                           for m in paper_metrics], rotation=0)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name} - CEM-MitoLab Paper Metrics (Table 2 Comparison)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, paper_means, paper_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add note
        ax.text(0.5, -0.15, 'Note: Metrics from CEM-MitoLab paper (Conrad & Narayan, 2023) Table 2',
               ha='center', transform=ax.transAxes, fontsize=10, style='italic')

        plt.tight_layout()
        plot_path = self.output_dir / 'paper_metrics_summary.png'
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
        print(f"{self.model_name.upper()} VALIDATION SET PERFORMANCE")
        print("=" * 60)
        print(f"Samples: {stats['n_samples']}")
        print(f"\n--- Standard Metrics ---")
        print(f"Dice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}")
        print(f"  Median: {stats['dice_median']:.4f}")
        print(f"  Range: [{stats['dice_min']:.4f}, {stats['dice_max']:.4f}]")
        print(f"\nIoU:              {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
        print(f"Pixel Accuracy:   {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}")
        print(f"Precision:        {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
        print(f"Recall:           {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")
        print(f"\n--- CEM-MitoLab Paper Metrics (Table 2) ---")
        print(f"Mean F1@50:       {stats['f1_50_mean']:.4f} ± {stats['f1_50_std']:.4f}")
        print(f"Mean F1@75:       {stats['f1_75_mean']:.4f} ± {stats['f1_75_std']:.4f}")
        print(f"Mean AP@50:       {stats['ap_50_mean']:.4f} ± {stats['ap_50_std']:.4f}")
        print(f"Mean AP@75:       {stats['ap_75_mean']:.4f} ± {stats['ap_75_std']:.4f}")
        print(f"Mean PQ:          {stats['pq_mean']:.4f} ± {stats['pq_std']:.4f}")

        # Save text report
        report_path = self.output_dir / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"{self.model_name.upper()} VALIDATION SET BENCHMARK\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Model Name: {self.model_name}\n")
            f.write(f"Validation samples: {stats['n_samples']}\n\n")

            f.write("STANDARD SEGMENTATION METRICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Dice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}\n")
            f.write(f"  Median: {stats['dice_median']:.4f}\n")
            f.write(f"  Range: [{stats['dice_min']:.4f}, {stats['dice_max']:.4f}]\n\n")
            f.write(f"IoU:              {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}\n")
            f.write(f"Pixel Accuracy:   {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}\n")
            f.write(f"Precision:        {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}\n")
            f.write(f"Recall:           {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}\n\n")

            f.write("CEM-MITOLAB PAPER METRICS (Table 2)\n")
            f.write("-" * 60 + "\n")
            f.write("Reference: Conrad & Narayan (2023)\n")
            f.write("'CEM-MitoLab: A domain-adaptive self-supervised method\n")
            f.write("for mitochondria segmentation'\n\n")
            f.write(f"Mean F1@50:       {stats['f1_50_mean']:.4f} ± {stats['f1_50_std']:.4f}\n")
            f.write(f"Mean F1@75:       {stats['f1_75_mean']:.4f} ± {stats['f1_75_std']:.4f}\n")
            f.write(f"Mean AP@50:       {stats['ap_50_mean']:.4f} ± {stats['ap_50_std']:.4f}\n")
            f.write(f"Mean AP@75:       {stats['ap_75_mean']:.4f} ± {stats['ap_75_std']:.4f}\n")
            f.write(f"Mean PQ:          {stats['pq_mean']:.4f} ± {stats['pq_std']:.4f}\n")
            f.write(f"  SQ (Seg Quality): {stats['sq_mean']:.4f} ± {stats['sq_std']:.4f}\n")
            f.write(f"  RQ (Rec Quality): {stats['rq_mean']:.4f} ± {stats['rq_std']:.4f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("NOTES:\n")
            f.write("- Benchmarked on validation set (20%) from seed=123 split\n")
            f.write("- This data was NOT used during training\n")
            f.write("- Paper metrics are instance-based (object-level matching)\n")
            f.write("- Standard metrics are pixel-based (semantic segmentation)\n")
            f.write("=" * 60 + "\n")

        print(f"Saved: {report_path}")

        return stats


def main():
    """Benchmark OurResUNet on validation set"""
    print("\n" + "=" * 60)
    print("OURRESUNET VALIDATION BENCHMARK")
    print("=" * 60)
    print("\nThis script benchmarks OurResUNet on the VALIDATION SET")
    print("Using the same seed=123 split from training")
    print("Ensures we test on data NOT used for training\n")

    # ========================================
    # PRE-CONFIGURED FOR OURRESUNET
    # ========================================

    MODEL_PATH = "../model/ourresunet_best.keras"  # Update if your model path differs
    MODEL_NAME = "OurResUNet"

    NPY_PATH = "../npydata"
    OUTPUT_DIR = "./benchmark_results_ourresunet"
    MAX_SAMPLES = 6000  # Must match training
    VAL_RATIO = 0.2     # Must match training

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print()

    # Initialize benchmark
    benchmark = ValidationBenchmark(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
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
    print(f"\nKey Results for {MODEL_NAME}:")
    print(f"  Dice (standard):       {stats['dice_mean']:.4f}")
    print(f"  Mean IoU (standard):   {stats['iou_mean']:.4f}")
    print(f"  Mean F1@50 (paper):    {stats['f1_50_mean']:.4f}")
    print(f"  Mean F1@75 (paper):    {stats['f1_75_mean']:.4f}")
    print(f"  Mean PQ (paper):       {stats['pq_mean']:.4f}")
    print("\nNote: Paper metrics match CEM-MitoLab (Conrad & Narayan, 2023) Table 2")


if __name__ == "__main__":
    main()
