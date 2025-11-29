import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import json
from datetime import datetime
from scipy import stats
from skimage import measure
from scipy.stats import pearsonr

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class SegmentationMetrics:
    """Calculate segmentation performance metrics"""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
        """
        Calculate Dice coefficient (F1 score for segmentation)
        
        Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()
        
        intersection = np.sum(pred & gt)
        dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
        
        return float(dice)
    
    @staticmethod
    def miou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
        """
        Calculate mean Intersection over Union (mIoU)
        
        IoU = |pred ∩ gt| / |pred ∪ gt|
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()
        
        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)
        
        iou = (intersection + smooth) / (union + smooth)
        
        return float(iou)
    
    @staticmethod
    def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Pixel Accuracy
        
        PA = (TP + TN) / (TP + TN + FP + FN)
        """
        pred = pred.astype(bool).flatten()
        gt = gt.astype(bool).flatten()
        
        correct = np.sum(pred == gt)
        total = len(pred)
        
        pa = correct / total
        
        return float(pa)


class MorphologicalFeatures:
    """Extract morphological features from mitochondrial masks"""
    
    @staticmethod
    def extract_features(mask: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from a binary mask
        
        Features extracted:
        - mean_area: Average area of mitochondria instances
        - total_area: Total area of all mitochondria
        - count: Number of mitochondria instances
        - mean_perimeter: Average perimeter
        - mean_eccentricity: Average eccentricity (elongation)
        - form_factor: Measure of circularity (4π*area/perimeter²)
        - aspect_ratio: Major axis / minor axis
        - solidity: Area / convex hull area
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Dictionary of morphological features
        """
        # Label connected components (individual mitochondria)
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) == 0:
            return {
                'mean_area': 0.0,
                'total_area': 0.0,
                'count': 0,
                'mean_perimeter': 0.0,
                'mean_eccentricity': 0.0,
                'mean_form_factor': 0.0,
                'mean_aspect_ratio': 0.0,
                'mean_solidity': 0.0,
            }
        
        # Extract features for each region
        areas = []
        perimeters = []
        eccentricities = []
        form_factors = []
        aspect_ratios = []
        solidities = []
        
        for region in regions:
            # Area
            area = region.area
            areas.append(area)
            
            # Perimeter
            perimeter = region.perimeter
            perimeters.append(perimeter)
            
            # Eccentricity (0 = circle, 1 = line)
            eccentricity = region.eccentricity
            eccentricities.append(eccentricity)
            
            # Form factor (circularity)
            if perimeter > 0:
                form_factor = (4 * np.pi * area) / (perimeter ** 2)
                form_factors.append(form_factor)
            
            # Aspect ratio (elongation)
            if region.minor_axis_length > 0:
                aspect_ratio = region.major_axis_length / region.minor_axis_length
                aspect_ratios.append(aspect_ratio)
            
            # Solidity (compactness)
            solidity = region.solidity
            solidities.append(solidity)
        
        return {
            'mean_area': float(np.mean(areas)),
            'total_area': float(np.sum(areas)),
            'count': len(regions),
            'mean_perimeter': float(np.mean(perimeters)),
            'mean_eccentricity': float(np.mean(eccentricities)),
            'mean_form_factor': float(np.mean(form_factors)) if form_factors else 0.0,
            'mean_aspect_ratio': float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            'mean_solidity': float(np.mean(solidities)),
        }


class OtsuSegmentation:
    """Otsu's thresholding method for mitochondrial segmentation"""
    
    def __init__(self, blur_kernel_size: int = 5):
        """
        Initialize Otsu segmentation
        
        Args:
            blur_kernel_size: Size of Gaussian blur kernel for preprocessing
        """
        self.blur_kernel_size = blur_kernel_size
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform Otsu's thresholding segmentation
        
        Args:
            image: Input grayscale image (H, W)
            
        Returns:
            Binary segmentation mask (H, W) with values 0 or 1
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        if self.blur_kernel_size > 0:
            blurred = cv2.GaussianBlur(image, (self.blur_kernel_size, self.blur_kernel_size), 0)
        else:
            blurred = image
        
        # Apply Otsu's thresholding
        _, binary_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary_mask.astype(np.uint8)


class MitochondriaDataset:
    """Handler for mitochondria dataset loading"""
    
    def __init__(self, root_path: str):
        """
        Initialize dataset
        
        Args:
            root_path: Path to root directory containing dataset folders
        """
        self.root_path = Path(root_path)
        self.datasets = self._scan_datasets()
    
    def _scan_datasets(self) -> List[Dict]:
        """Scan for all valid dataset folders"""
        datasets = []
        
        for item in self.root_path.iterdir():
            if not item.is_dir():
                continue
            
            images_dir = item / 'images'
            masks_dir = item / 'masks'
            
            if images_dir.exists() and masks_dir.exists():
                # Get all image files
                image_files = sorted(list(images_dir.glob('*.tif*')))
                mask_files = sorted(list(masks_dir.glob('*.tif*')))
                
                if len(image_files) > 0 and len(image_files) == len(mask_files):
                    datasets.append({
                        'name': item.name,
                        'path': item,
                        'images': image_files,
                        'masks': mask_files,
                        'n_samples': len(image_files)
                    })
        
        return datasets
    
    def get_sample(self, dataset_idx: int, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Get a single image-mask pair
        
        Returns:
            image, mask, image_path, mask_path
        """
        dataset = self.datasets[dataset_idx]
        
        image_path = dataset['images'][sample_idx]
        mask_path = dataset['masks'][sample_idx]
        
        # Load image and mask
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        # Convert instance mask to binary (any non-zero value becomes 1)
        mask_binary = (mask > 0).astype(np.uint8)
        
        return image, mask_binary, str(image_path), str(mask_path)
    
    def __len__(self):
        return len(self.datasets)
    
    def total_samples(self):
        return sum(d['n_samples'] for d in self.datasets)


class BenchmarkRunner:
    """Run benchmark evaluation on segmentation algorithms"""
    
    def __init__(self, data_root: str, output_dir: str):
        """
        Initialize benchmark runner
        
        Args:
            data_root: Path to dataset root
            output_dir: Path to save outputs
        """
        self.dataset = MitochondriaDataset(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        
    def run_otsu_benchmark(self, max_samples: int = None, sample_datasets: int = None):
        """
        Run Otsu's thresholding benchmark
        
        Args:
            max_samples: Maximum number of samples to process (None = all)
            sample_datasets: Number of datasets to sample (None = all)
        """
        print("="*80)
        print("OTSU'S THRESHOLDING BENCHMARK")
        print("="*80)
        print(f"Total datasets: {len(self.dataset)}")
        print(f"Total samples: {self.dataset.total_samples()}")
        
        # Initialize segmentation method
        otsu = OtsuSegmentation(blur_kernel_size=5)
        metrics_calc = SegmentationMetrics()
        morph_features = MorphologicalFeatures()
        
        # Determine which datasets to process
        if sample_datasets is not None:
            dataset_indices = np.random.choice(
                len(self.dataset), 
                min(sample_datasets, len(self.dataset)), 
                replace=False
            )
        else:
            dataset_indices = range(len(self.dataset))
        
        # Track samples processed
        samples_processed = 0
        
        # Process each dataset
        for dataset_idx in tqdm(dataset_indices, desc="Processing datasets"):
            dataset_info = self.dataset.datasets[dataset_idx]
            
            # Process each sample in dataset
            for sample_idx in range(dataset_info['n_samples']):
                if max_samples is not None and samples_processed >= max_samples:
                    break
                
                # Load data
                image, gt_mask, img_path, mask_path = self.dataset.get_sample(
                    dataset_idx, sample_idx
                )
                
                # Run segmentation
                pred_mask = otsu.segment(image)
                
                # Calculate metrics
                dice = metrics_calc.dice_coefficient(pred_mask, gt_mask)
                miou = metrics_calc.miou(pred_mask, gt_mask)
                pa = metrics_calc.pixel_accuracy(pred_mask, gt_mask)
                
                # Extract morphological features from prediction and ground truth
                pred_features = morph_features.extract_features(pred_mask)
                gt_features = morph_features.extract_features(gt_mask)
                
                # Store results
                result = {
                    'dataset': dataset_info['name'],
                    'sample_idx': sample_idx,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'dice': dice,
                    'miou': miou,
                    'pa': pa,
                    'image_shape': image.shape,
                }
                
                # Add predicted morphological features
                for key, value in pred_features.items():
                    result[f'pred_{key}'] = value
                
                # Add ground truth morphological features
                for key, value in gt_features.items():
                    result[f'gt_{key}'] = value
                
                self.results.append(result)
                
                samples_processed += 1
            
            if max_samples is not None and samples_processed >= max_samples:
                break
        
        print(f"\n✓ Processed {samples_processed} samples from {len(set(r['dataset'] for r in self.results))} datasets")
        
        return self.results
    
    def calculate_statistics(self) -> Dict:
        """Calculate mean and std for all metrics"""
        df = pd.DataFrame(self.results)
        
        stats = {
            'dice_mean': df['dice'].mean(),
            'dice_std': df['dice'].std(),
            'miou_mean': df['miou'].mean(),
            'miou_std': df['miou'].std(),
            'pa_mean': df['pa'].mean(),
            'pa_std': df['pa'].std(),
            'n_samples': len(df),
            'n_datasets': df['dataset'].nunique(),
        }
        
        return stats
    
    def calculate_feature_correlations(self) -> Dict:
        """
        Calculate Pearson correlation between predicted and ground truth morphological features
        Returns correlations and p-values for statistical significance
        """
        df = pd.DataFrame(self.results)
        
        # Morphological features to analyze
        features = ['mean_area', 'count', 'mean_perimeter', 'mean_eccentricity', 
                   'mean_form_factor', 'mean_aspect_ratio', 'mean_solidity']
        
        correlations = {}
        
        for feature in features:
            pred_col = f'pred_{feature}'
            gt_col = f'gt_{feature}'
            
            if pred_col in df.columns and gt_col in df.columns:
                # Remove any NaN or inf values
                valid_mask = np.isfinite(df[pred_col]) & np.isfinite(df[gt_col])
                pred_values = df[pred_col][valid_mask].values
                gt_values = df[gt_col][valid_mask].values
                
                if len(pred_values) > 1:
                    # Calculate Pearson correlation and p-value
                    corr, p_value = pearsonr(pred_values, gt_values)
                    
                    correlations[feature] = {
                        'pearson_r': float(corr),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05),
                        'n_samples': int(len(pred_values))
                    }
        
        return correlations
    
    def save_results(self):
        """Save all results to CSV and JSON"""
        # Save detailed results
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / 'otsu_detailed_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved detailed results to: {csv_path}")
        
        # Save summary statistics
        stats = self.calculate_statistics()
        
        stats_path = self.output_dir / 'otsu_summary_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved summary statistics to: {stats_path}")
        
        # Calculate and save feature correlations
        correlations = self.calculate_feature_correlations()
        
        corr_path = self.output_dir / 'otsu_feature_correlations.json'
        with open(corr_path, 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"✓ Saved feature correlations to: {corr_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Samples processed: {stats['n_samples']}")
        print(f"Datasets: {stats['n_datasets']}")
        print(f"\nDice Coefficient: {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}")
        print(f"mIoU:             {stats['miou_mean']:.4f} ± {stats['miou_std']:.4f}")
        print(f"Pixel Accuracy:   {stats['pa_mean']:.4f} ± {stats['pa_std']:.4f}")
        
        # Print feature correlations
        print("\n" + "="*80)
        print("MORPHOLOGICAL FEATURE CORRELATIONS (Pearson)")
        print("="*80)
        for feature, corr_data in correlations.items():
            sig_marker = "***" if corr_data['p_value'] < 0.001 else \
                        "**" if corr_data['p_value'] < 0.01 else \
                        "*" if corr_data['p_value'] < 0.05 else "ns"
            print(f"{feature:20s}: r={corr_data['pearson_r']:7.4f}, "
                  f"p={corr_data['p_value']:.4e} {sig_marker}")
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
        
        return stats
    
    def plot_metrics(self):
        """Create visualization of metrics"""
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram of Dice scores
        axes[0, 0].hist(df['dice'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(df['dice'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["dice"].mean():.3f}')
        axes[0, 0].set_xlabel('Dice Coefficient')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Dice Coefficient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of mIoU scores
        axes[0, 1].hist(df['miou'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].axvline(df['miou'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["miou"].mean():.3f}')
        axes[0, 1].set_xlabel('mIoU')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of mIoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram of PA scores
        axes[1, 0].hist(df['pa'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[1, 0].axvline(df['pa'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["pa"].mean():.3f}')
        axes[1, 0].set_xlabel('Pixel Accuracy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Pixel Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scatter plot - Dice vs mIoU
        axes[1, 1].scatter(df['dice'], df['miou'], alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Dice Coefficient')
        axes[1, 1].set_ylabel('mIoU')
        axes[1, 1].set_title('Dice vs mIoU Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df['dice'].corr(df['miou'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'otsu_metrics_distributions.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics visualization to: {fig_path}")
        plt.close()
    
    def plot_comparison_bar(self):
        """Create bar plot comparing metrics (for future multi-algorithm comparison)"""
        stats = self.calculate_statistics()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ['Dice', 'mIoU', 'PA']
        means = [stats['dice_mean'], stats['miou_mean'], stats['pa_mean']]
        stds = [stats['dice_std'], stats['miou_std'], stats['pa_std']]
        
        x = np.arange(len(metrics))
        width = 0.6
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                     color='orange', alpha=0.7, label='Otsu')
        
        ax.set_ylabel('Score')
        ax.set_title('Otsu Thresholding Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'otsu_performance_bars.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance comparison to: {fig_path}")
        plt.close()
    
    def plot_feature_correlations(self):
        """Create scatter plots showing correlation between predicted and ground truth features"""
        df = pd.DataFrame(self.results)
        correlations = self.calculate_feature_correlations()
        
        # Select features to plot
        features = ['mean_area', 'count', 'mean_eccentricity', 'mean_form_factor']
        n_features = len(features)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            pred_col = f'pred_{feature}'
            gt_col = f'gt_{feature}'
            
            if pred_col in df.columns and gt_col in df.columns:
                # Get data
                valid_mask = np.isfinite(df[pred_col]) & np.isfinite(df[gt_col])
                pred_values = df[pred_col][valid_mask]
                gt_values = df[gt_col][valid_mask]
                
                # Scatter plot
                axes[idx].scatter(gt_values, pred_values, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
                
                # Add identity line (perfect prediction)
                min_val = min(gt_values.min(), pred_values.min())
                max_val = max(gt_values.max(), pred_values.max())
                axes[idx].plot([min_val, max_val], [min_val, max_val], 
                             'r--', linewidth=2, label='Perfect prediction')
                
                # Add regression line
                if len(gt_values) > 1:
                    z = np.polyfit(gt_values, pred_values, 1)
                    p = np.poly1d(z)
                    axes[idx].plot(gt_values, p(gt_values), 'b-', alpha=0.7, linewidth=2, label='Fit')
                
                # Labels and title
                axes[idx].set_xlabel(f'Ground Truth {feature.replace("_", " ").title()}', fontsize=10)
                axes[idx].set_ylabel(f'Predicted {feature.replace("_", " ").title()}', fontsize=10)
                
                # Add correlation info
                if feature in correlations:
                    corr_data = correlations[feature]
                    r = corr_data['pearson_r']
                    p = corr_data['p_value']
                    sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    
                    title = f'{feature.replace("_", " ").title()}\n'
                    title += f'r = {r:.3f} {sig_marker}, p = {p:.4e}'
                    axes[idx].set_title(title, fontsize=11)
                
                axes[idx].legend(loc='upper left', fontsize=8)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'otsu_feature_correlations.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature correlation plots to: {fig_path}")
        plt.close()
    
    def visualize_samples(self, n_samples: int = 6):
        """Visualize sample segmentations"""
        # Select random samples
        sample_indices = np.random.choice(len(self.results), min(n_samples, len(self.results)), replace=False)
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        otsu = OtsuSegmentation(blur_kernel_size=5)
        
        for idx, result_idx in enumerate(sample_indices):
            result = self.results[result_idx]
            
            # Load image and mask
            image = np.array(Image.open(result['image_path']))
            gt_mask = np.array(Image.open(result['mask_path']))
            gt_mask_binary = (gt_mask > 0).astype(np.uint8)
            
            # Generate prediction
            pred_mask = otsu.segment(image)
            
            # Plot original image
            axes[idx, 0].imshow(image, cmap='gray')
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            # Plot ground truth
            axes[idx, 1].imshow(gt_mask_binary, cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Plot prediction
            axes[idx, 2].imshow(pred_mask, cmap='gray')
            axes[idx, 2].set_title(f'Otsu Prediction\nDice: {result["dice"]:.3f}')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'otsu_sample_segmentations.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved sample visualizations to: {fig_path}")
        plt.close()


def main():
    """Main execution function"""
    
    # Configuration
    DATA_ROOT = "../../data/cem_mitolab"
    OUTPUT_DIR = "outputs"
    
    # Processing options
    MAX_SAMPLES = 200  # Set to None to process all samples
    SAMPLE_DATASETS = None  # Set to None to use all datasets
    
    print("\n" + "="*80)
    print("MITOCHONDRIAL SEGMENTATION BENCHMARK - OTSU'S THRESHOLDING")
    print("="*80)
    print(f"Data root: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Initialize benchmark
    benchmark = BenchmarkRunner(DATA_ROOT, OUTPUT_DIR)
    
    # Run benchmark
    print("\nRunning Otsu's thresholding benchmark...")
    results = benchmark.run_otsu_benchmark(
        max_samples=MAX_SAMPLES,
        sample_datasets=SAMPLE_DATASETS
    )
    
    # Save results
    print("\nSaving results...")
    stats = benchmark.save_results()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    benchmark.plot_metrics()
    benchmark.plot_comparison_bar()
    benchmark.plot_feature_correlations()
    benchmark.visualize_samples(n_samples=6)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - otsu_detailed_results.csv (all samples with features)")
    print("  - otsu_summary_stats.json (mean ± SD metrics)")
    print("  - otsu_feature_correlations.json (Pearson r & p-values)")
    print("  - otsu_metrics_distributions.png (histogram plots)")
    print("  - otsu_performance_bars.png (bar chart)")
    print("  - otsu_feature_correlations.png (scatter plots with r & p)")
    print("  - otsu_sample_segmentations.png (visual examples)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()