"""
Benchmarking script for MitoSegNet
Evaluates model performance with comprehensive metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from skimage import measure
import json
from datetime import datetime
import glob


# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class SegmentationMetrics:
    """Calculate segmentation performance metrics"""

    @staticmethod
    def dice_coefficient(pred, gt, smooth=1e-6):
        """
        Calculate Dice coefficient (F1 score)

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            smooth: Smoothing factor

        Returns:
            Dice coefficient value
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()

        intersection = np.sum(pred & gt)
        dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

        return float(dice)

    @staticmethod
    def iou(pred, gt, smooth=1e-6):
        """
        Calculate Intersection over Union (IoU)

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            smooth: Smoothing factor

        Returns:
            IoU value
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()

        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)

        iou = (intersection + smooth) / (union + smooth)

        return float(iou)

    @staticmethod
    def pixel_accuracy(pred, gt):
        """
        Calculate pixel-wise accuracy

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask

        Returns:
            Pixel accuracy value
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()

        correct = np.sum(pred == gt)
        total = len(pred)

        return float(correct / total)

    @staticmethod
    def precision_recall(pred, gt, smooth=1e-6):
        """
        Calculate precision and recall

        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            smooth: Smoothing factor

        Returns:
            tuple: (precision, recall)
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()

        true_positive = np.sum(pred & gt)
        false_positive = np.sum(pred & ~gt)
        false_negative = np.sum(~pred & gt)

        precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
        recall = (true_positive + smooth) / (true_positive + false_negative + smooth)

        return float(precision), float(recall)


class MorphologicalFeatures:
    """Extract morphological features from masks"""

    @staticmethod
    def extract_features(mask):
        """
        Extract morphological features from binary mask

        Args:
            mask: Binary mask

        Returns:
            Dictionary of features
        """
        # Label connected components
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)

        if len(regions) == 0:
            return {
                'count': 0,
                'mean_area': 0.0,
                'total_area': 0.0,
                'mean_perimeter': 0.0,
                'mean_eccentricity': 0.0,
                'mean_solidity': 0.0,
            }

        areas = [r.area for r in regions]
        perimeters = [r.perimeter for r in regions]
        eccentricities = [r.eccentricity for r in regions]
        solidities = [r.solidity for r in regions]

        return {
            'count': len(regions),
            'mean_area': float(np.mean(areas)),
            'total_area': float(np.sum(areas)),
            'mean_perimeter': float(np.mean(perimeters)),
            'mean_eccentricity': float(np.mean(eccentricities)),
            'mean_solidity': float(np.mean(solidities)),
        }


class BenchmarkRunner:
    """Run comprehensive benchmarking evaluation"""

    def __init__(self, gt_dir, pred_dir, output_dir="./benchmark_results"):
        """
        Initialize benchmark runner

        Args:
            gt_dir: Directory containing ground truth masks
            pred_dir: Directory containing predicted masks
            output_dir: Directory to save results
        """
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.metrics_calc = SegmentationMetrics()
        self.morph_calc = MorphologicalFeatures()

    def run_evaluation(self, max_samples=None):
        """
        Run full evaluation

        Args:
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            List of results dictionaries
        """
        print("=" * 60)
        print("RUNNING BENCHMARK EVALUATION")
        print("=" * 60)
        print(f"Ground truth: {self.gt_dir}")
        print(f"Predictions: {self.pred_dir}")
        print(f"Output: {self.output_dir}")

        # Get matching image pairs
        gt_files = sorted(list(self.gt_dir.glob("*.tif*")))
        print(f"\nFound {len(gt_files)} ground truth images")

        if max_samples is not None:
            gt_files = gt_files[:max_samples]
            print(f"Evaluating first {max_samples} images")

        # Evaluate each pair
        for gt_path in tqdm(gt_files, desc="Evaluating"):
            # Load ground truth
            gt_mask = np.array(Image.open(gt_path))
            gt_binary = (gt_mask > 0).astype(np.uint8)

            # Find matching prediction
            pred_path = self.pred_dir / gt_path.name
            if not pred_path.exists():
                print(f"Warning: No prediction for {gt_path.name}, skipping")
                continue

            # Load prediction
            pred_mask = np.array(Image.open(pred_path))
            pred_binary = (pred_mask > 0).astype(np.uint8)

            # Calculate segmentation metrics
            dice = self.metrics_calc.dice_coefficient(pred_binary, gt_binary)
            iou = self.metrics_calc.iou(pred_binary, gt_binary)
            pa = self.metrics_calc.pixel_accuracy(pred_binary, gt_binary)
            precision, recall = self.metrics_calc.precision_recall(pred_binary, gt_binary)

            # Calculate morphological features
            pred_features = self.morph_calc.extract_features(pred_binary)
            gt_features = self.morph_calc.extract_features(gt_binary)

            # Store results
            result = {
                'image': gt_path.name,
                'dice': dice,
                'iou': iou,
                'pixel_accuracy': pa,
                'precision': precision,
                'recall': recall,
            }

            # Add morphological features
            for key, value in pred_features.items():
                result[f'pred_{key}'] = value
            for key, value in gt_features.items():
                result[f'gt_{key}'] = value

            self.results.append(result)

        print(f"\nEvaluated {len(self.results)} image pairs")

        return self.results

    def calculate_statistics(self):
        """Calculate summary statistics"""
        df = pd.DataFrame(self.results)

        stats = {
            'n_samples': len(df),
            'dice_mean': float(df['dice'].mean()),
            'dice_std': float(df['dice'].std()),
            'dice_median': float(df['dice'].median()),
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

    def save_results(self):
        """Save all results and statistics"""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        # Save detailed results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'detailed_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed results: {csv_path}")

        # Calculate and save statistics
        stats = self.calculate_statistics()
        stats_path = self.output_dir / 'summary_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved summary statistics: {stats_path}")

        # Print statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Samples: {stats['n_samples']}")
        print(f"\nDice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}")
        print(f"IoU:              {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
        print(f"Pixel Accuracy:   {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}")
        print(f"Precision:        {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
        print(f"Recall:           {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")

        return stats

    def plot_metrics(self):
        """Create metric distribution plots"""
        print("\nGenerating metric plots...")

        df = pd.DataFrame(self.results)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics = ['dice', 'iou', 'pixel_accuracy', 'precision', 'recall']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'peachpuff']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in df.columns:
                # Histogram
                axes[idx].hist(df[metric], bins=30, edgecolor='black', alpha=0.7, color=color)
                axes[idx].axvline(df[metric].mean(), color='red', linestyle='--',
                                linewidth=2, label=f'Mean: {df[metric].mean():.3f}')
                axes[idx].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
                axes[idx].set_ylabel('Frequency', fontsize=12)
                axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
                axes[idx].legend(fontsize=10)
                axes[idx].grid(True, alpha=0.3)

        # Dice vs IoU scatter
        axes[5].scatter(df['dice'], df['iou'], alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[5].set_xlabel('Dice Coefficient', fontsize=12)
        axes[5].set_ylabel('IoU', fontsize=12)
        axes[5].set_title('Dice vs IoU', fontsize=14, fontweight='bold')
        axes[5].grid(True, alpha=0.3)

        # Calculate correlation
        if len(df) > 1:
            corr = df['dice'].corr(df['iou'])
            axes[5].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=axes[5].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save
        plot_path = self.output_dir / 'metrics_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

    def plot_comparison_bars(self):
        """Create bar plot of metrics"""
        print("Generating comparison bar plot...")

        stats = self.calculate_statistics()

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['Dice', 'IoU', 'Pixel Acc', 'Precision', 'Recall']
        means = [
            stats['dice_mean'],
            stats['iou_mean'],
            stats['pixel_accuracy_mean'],
            stats['precision_mean'],
            stats['recall_mean']
        ]
        stds = [
            stats['dice_std'],
            stats['iou_std'],
            stats['pixel_accuracy_std'],
            stats['precision_std'],
            stats['recall_std']
        ]

        x = np.arange(len(metrics))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'],
                     edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('MitoSegNet Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()

        # Save
        plot_path = self.output_dir / 'performance_bars.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

    def create_report(self):
        """Create comprehensive text report"""
        print("Generating text report...")

        stats = self.calculate_statistics()
        report_path = self.output_dir / 'benchmark_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MITOSEGNET BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Ground Truth: {self.gt_dir}\n")
            f.write(f"Predictions: {self.pred_dir}\n\n")

            f.write("=" * 60 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Number of Samples: {stats['n_samples']}\n\n")

            f.write("Segmentation Metrics:\n")
            f.write(f"  Dice Coefficient:  {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}\n")
            f.write(f"  IoU:               {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}\n")
            f.write(f"  Pixel Accuracy:    {stats['pixel_accuracy_mean']:.4f} ± {stats['pixel_accuracy_std']:.4f}\n")
            f.write(f"  Precision:         {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}\n")
            f.write(f"  Recall:            {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}\n\n")

            f.write("Median Values:\n")
            f.write(f"  Dice (median):     {stats['dice_median']:.4f}\n")
            f.write(f"  IoU (median):      {stats['iou_median']:.4f}\n\n")

        print(f"Saved: {report_path}")


def main():
    """Main benchmarking function"""
    print("\n" + "=" * 60)
    print("MITOSEGNET BENCHMARKING")
    print("=" * 60)

    # Configuration
    GT_DIR = "../test/masks"  # Ground truth masks
    PRED_DIR = "./outputs/trained_predictions/binary"  # Predictions from trained model
    OUTPUT_DIR = "./benchmark_results"
    MAX_SAMPLES = None  # None for all samples

    print(f"\nConfiguration:")
    print(f"  Ground truth: {GT_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")

    # Initialize benchmark runner
    benchmark = BenchmarkRunner(GT_DIR, PRED_DIR, OUTPUT_DIR)

    # Run evaluation
    results = benchmark.run_evaluation(max_samples=MAX_SAMPLES)

    if len(results) == 0:
        print("\nNo results to process. Check that prediction and ground truth directories match.")
        return

    # Save results
    benchmark.save_results()

    # Create visualizations
    benchmark.plot_metrics()
    benchmark.plot_comparison_bars()

    # Create report
    benchmark.create_report()

    print("\n" + "=" * 60)
    print("BENCHMARKING COMPLETE!")
    print("=" * 60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()