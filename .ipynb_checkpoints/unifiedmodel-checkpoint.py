"""
Unified Segmentation and Density Classification Model - Standalone Version

This file contains the unified models with all dependencies included.
Performs both signal-background segmentation and intensity density classification in one step.

Usage:
    from unified_model_standalone import UnifiedSegmentationDensityModel
    
    model = UnifiedSegmentationDensityModel(n_density_classes=3)
    signal_mask, density_class, stats = model.process(image)
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional, Union, Dict
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Deep learning imports (optional, for U-Net model)
try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, UpSampling2D,
        Concatenate, Conv2DTranspose, BatchNormalization
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not available. U-Net model will not work.")


# ============================================================================
# DEPENDENCY CLASSES
# ============================================================================

class InverseThresholdModel:
    """
    Simple thresholding-based model for separating dark signal from bright background.
    Used internally by the unified model.
    """
    
    def __init__(self, method='otsu', blur_size=5, adaptive_block_size=11, adaptive_c=2):
        """
        Initialize thresholding model
        
        Args:
            method: 'otsu', 'adaptive', or 'percentile'
            blur_size: Gaussian blur kernel size (odd number, 0 to disable)
            adaptive_block_size: Block size for adaptive thresholding (odd number)
            adaptive_c: Constant subtracted from mean in adaptive thresholding
        """
        self.method = method
        self.blur_size = blur_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: ensure uint8, apply blur if needed"""
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        if self.blur_size > 0:
            image = cv2.GaussianBlur(image, (self.blur_size, self.blur_size), 0)
        
        return image
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image: dark regions (signal) vs bright regions (background)
        
        Args:
            image: Input grayscale image (H, W) or (H, W, C)
            
        Returns:
            Binary mask: 1 = signal (dark), 0 = background (bright)
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = np.mean(image, axis=2)
        
        # Preprocess
        image = self.preprocess(image)
        
        # Apply thresholding method
        if self.method == 'otsu':
            # Otsu's method: automatically finds optimal threshold
            threshold_value, binary = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # THRESH_BINARY_INV: dark pixels (low intensity) become 1
            
        elif self.method == 'adaptive':
            # Adaptive thresholding: handles varying illumination
            binary = cv2.adaptiveThreshold(
                image, 1, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_block_size,
                self.adaptive_c
            )
            
        elif self.method == 'percentile':
            # Percentile-based thresholding
            threshold_value = np.percentile(image, 30)  # Bottom 30% = signal
            binary = (image < threshold_value).astype(np.uint8)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return binary.astype(np.uint8)


class IntensityDensityClassifier:
    """
    Classifies different intensity densities within low-intensity signal regions.
    Used internally by the unified model.
    """
    
    def __init__(self, n_classes=3, method='percentile', blur_size=3):
        """
        Initialize density classifier
        
        Args:
            n_classes: Number of density classes (e.g., 3 = low/medium/high density)
            method: 'percentile', 'kmeans', or 'equal_width'
            blur_size: Gaussian blur kernel size for smoothing (odd number, 0 to disable)
        """
        self.n_classes = n_classes
        self.method = method
        self.blur_size = blur_size
        self.thresholds = None
    
    def preprocess(self, image: np.ndarray, signal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image and mask: extract signal regions only"""
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure binary mask
        if signal_mask.dtype != np.uint8:
            signal_mask = (signal_mask > 0.5).astype(np.uint8)
        
        # Apply blur to reduce noise
        if self.blur_size > 0:
            image = cv2.GaussianBlur(image, (self.blur_size, self.blur_size), 0)
        
        return image, signal_mask
    
    def compute_thresholds(self, image: np.ndarray, signal_mask: np.ndarray) -> np.ndarray:
        """Compute intensity thresholds for density classification"""
        # Extract signal region intensities
        signal_intensities = image[signal_mask == 1]
        
        if len(signal_intensities) == 0:
            # No signal regions, return default thresholds
            return np.linspace(0, 255, self.n_classes + 1)[1:-1]
        
        if self.method == 'percentile':
            # Use percentiles to divide into equal-sized groups
            percentiles = np.linspace(0, 100, self.n_classes + 1)[1:-1]
            thresholds = np.percentile(signal_intensities, percentiles)
            
        elif self.method == 'equal_width':
            # Divide intensity range into equal-width bins
            min_intensity = signal_intensities.min()
            max_intensity = signal_intensities.max()
            thresholds = np.linspace(min_intensity, max_intensity, self.n_classes + 1)[1:-1]
            
        elif self.method == 'kmeans':
            # Use k-means clustering (simple implementation)
            try:
                from sklearn.cluster import KMeans
                intensities_reshaped = signal_intensities.reshape(-1, 1)
                kmeans = KMeans(n_clusters=self.n_classes, random_state=42, n_init=10)
                kmeans.fit(intensities_reshaped)
                centers = np.sort(kmeans.cluster_centers_.flatten())
                thresholds = (centers[:-1] + centers[1:]) / 2
            except ImportError:
                raise ImportError("scikit-learn required for kmeans method. Install with: pip install scikit-learn")
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return thresholds
    
    def classify(self, image: np.ndarray, signal_mask: np.ndarray) -> np.ndarray:
        """
        Classify signal regions into different density classes
        
        Args:
            image: Original grayscale image (H, W) or (H, W, C)
            signal_mask: Binary mask where 1 = signal regions (H, W)
            
        Returns:
            Multi-class mask: 0 = background, 1..n_classes = density classes
            (1 = lowest density, n_classes = highest density)
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = np.mean(image, axis=2)
        
        # Preprocess
        image, signal_mask = self.preprocess(image, signal_mask)
        
        # Compute thresholds
        thresholds = self.compute_thresholds(image, signal_mask)
        self.thresholds = thresholds
        
        # Initialize classification mask (0 = background)
        classification = np.zeros_like(image, dtype=np.uint8)
        
        # Classify signal regions
        signal_pixels = signal_mask == 1
        
        # Assign classes based on thresholds
        # Class 1: lowest density (darkest)
        classification[signal_pixels & (image <= thresholds[0])] = 1
        
        # Middle classes
        for i in range(1, len(thresholds)):
            classification[signal_pixels & (image > thresholds[i-1]) & (image <= thresholds[i])] = i + 1
        
        # Class n_classes: highest density (brightest signal)
        classification[signal_pixels & (image > thresholds[-1])] = self.n_classes
        
        return classification
    
    def classify_with_stats(self, image: np.ndarray, signal_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Classify and return statistics
        
        Args:
            image: Original grayscale image
            signal_mask: Binary mask where 1 = signal regions
            
        Returns:
            tuple: (classification_mask, statistics_dict)
        """
        classification = self.classify(image, signal_mask)
        
        # Compute statistics for each class
        stats = {
            'n_classes': self.n_classes,
            'thresholds': self.thresholds.tolist() if self.thresholds is not None else None,
            'class_counts': {},
            'class_intensities': {}
        }
        
        for class_id in range(1, self.n_classes + 1):
            class_mask = classification == class_id
            stats['class_counts'][class_id] = int(np.sum(class_mask))
            
            if np.sum(class_mask) > 0:
                class_intensities = image[class_mask]
                stats['class_intensities'][class_id] = {
                    'mean': float(np.mean(class_intensities)),
                    'std': float(np.std(class_intensities)),
                    'min': int(np.min(class_intensities)),
                    'max': int(np.max(class_intensities))
                }
        
        return classification, stats


# ============================================================================
# UNIFIED MODELS
# ============================================================================

class UnifiedSegmentationDensityModel:
    """
    Unified model that performs both segmentation and density classification in one step.
    Combines signal-background separation with intensity density classification.
    
    This is a thresholding-based model that requires no training.
    
    Example:
        model = UnifiedSegmentationDensityModel(n_density_classes=3)
        signal_mask, density_class, stats = model.process(image)
    """
    
    def __init__(self, 
                 seg_method='otsu',
                 density_method='percentile',
                 n_density_classes=3,
                 blur_size=5,
                 adaptive_block_size=11,
                 adaptive_c=2):
        """
        Initialize unified model
        
        Args:
            seg_method: Segmentation method ('otsu', 'adaptive', 'percentile')
            density_method: Density classification method ('percentile', 'equal_width', 'kmeans')
            n_density_classes: Number of density classes (default 3)
            blur_size: Gaussian blur kernel size
            adaptive_block_size: Block size for adaptive thresholding
            adaptive_c: Constant for adaptive thresholding
        """
        # Initialize segmentation model
        self.seg_model = InverseThresholdModel(
            method=seg_method,
            blur_size=blur_size,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c
        )
        
        # Initialize density classifier
        self.density_model = IntensityDensityClassifier(
            n_classes=n_density_classes,
            method=density_method,
            blur_size=blur_size
        )
        
        self.n_density_classes = n_density_classes
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Process image: segment and classify densities in one step
        
        Args:
            image: Input grayscale image (H, W) or (H, W, C)
            
        Returns:
            tuple: (signal_mask, density_classification, statistics)
            - signal_mask: Binary mask (1 = signal, 0 = background)
            - density_classification: Multi-class mask (0 = background, 1..n = density classes)
            - statistics: Dictionary with classification statistics
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = np.mean(image, axis=2)
        
        # Step 1: Segment signal from background
        signal_mask = self.seg_model.segment(image)
        
        # Step 2: Classify densities within signal regions
        density_classification, stats = self.density_model.classify_with_stats(
            image, signal_mask
        )
        
        return signal_mask, density_classification, stats
    
    def process_image(self, image_path: Union[str, Path],
                     output_signal_path: Optional[Union[str, Path]] = None,
                     output_density_path: Optional[Union[str, Path]] = None,
                     visualize: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Process image file
        
        Args:
            image_path: Path to input image
            output_signal_path: Optional path to save signal mask
            output_density_path: Optional path to save density classification
            visualize: Whether to show visualization
            
        Returns:
            tuple: (signal_mask, density_classification, statistics)
        """
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        image = np.array(Image.open(image_path).convert('L'))
        
        # Process
        signal_mask, density_classification, stats = self.process(image)
        
        # Save if requested
        if output_signal_path:
            if isinstance(output_signal_path, str):
                output_signal_path = Path(output_signal_path)
            output_signal_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(signal_mask * 255).save(output_signal_path)
        
        if output_density_path:
            if isinstance(output_density_path, str):
                output_density_path = Path(output_density_path)
            output_density_path.parent.mkdir(parents=True, exist_ok=True)
            # Scale density classes for visualization (0-255)
            density_vis = (density_classification * (255 // self.n_density_classes)).astype(np.uint8)
            Image.fromarray(density_vis).save(output_density_path)
        
        # Visualize if requested
        if visualize:
            self.visualize(image, signal_mask, density_classification, stats)
        
        return signal_mask, density_classification, stats
    
    def process_directory(self, input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         pattern: str = "*.tif*"):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            pattern: File pattern to match
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signal_dir = output_dir / "signal_masks"
        density_dir = output_dir / "density_classifications"
        signal_dir.mkdir(exist_ok=True)
        density_dir.mkdir(exist_ok=True)
        
        image_files = list(input_dir.glob(pattern))
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            signal_path = signal_dir / f"{img_path.stem}_signal{img_path.suffix}"
            density_path = density_dir / f"{img_path.stem}_density{img_path.suffix}"
            
            try:
                self.process_image(img_path, signal_path, density_path)
                print(f"Processed: {img_path.name}")
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
    
    def visualize(self, image: np.ndarray, signal_mask: np.ndarray,
                  density_classification: np.ndarray, stats: Optional[dict] = None):
        """Visualize segmentation and density classification"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Signal mask
        axes[0, 1].imshow(signal_mask, cmap='gray')
        axes[0, 1].set_title('Signal Segmentation\n(White = Signal, Black = Background)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Density classification (colored)
        cmap = plt.cm.get_cmap('viridis', self.n_density_classes)
        colored = cmap(density_classification / max(self.n_density_classes, 1))
        colored[density_classification == 0] = [0, 0, 0, 1]  # Black for background
        axes[0, 2].imshow(colored)
        axes[0, 2].set_title(f'Density Classification\n({self.n_density_classes} classes)', 
                           fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Overlay: Image + Density Classes
        overlay = image.copy().astype(np.float32)
        overlay = overlay / overlay.max()
        overlay_rgb = np.stack([overlay] * 3, axis=-1)
        
        for class_id in range(1, self.n_density_classes + 1):
            class_mask = density_classification == class_id
            color = cmap(class_id / self.n_density_classes)[:3]
            overlay_rgb[class_mask] = color
        
        axes[1, 0].imshow(overlay_rgb)
        axes[1, 0].set_title('Overlay: Image + Density Classes', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Statistics plot
        if stats:
            axes[1, 1].axis('off')
            stats_text = f"Classification Statistics\n\n"
            stats_text += f"Classes: {stats['n_classes']}\n"
            if stats['thresholds']:
                stats_text += f"Thresholds: {[f'{t:.1f}' for t in stats['thresholds']]}\n\n"
            
            for class_id in range(1, self.n_density_classes + 1):
                if class_id in stats['class_counts']:
                    count = stats['class_counts'][class_id]
                    pct = 100 * count / density_classification.size
                    stats_text += f"Class {class_id}: {count} pixels ({pct:.1f}%)\n"
                    if class_id in stats['class_intensities']:
                        int_stats = stats['class_intensities'][class_id]
                        stats_text += f"  Intensity: {int_stats['mean']:.1f} ± {int_stats['std']:.1f}\n"
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                           verticalalignment='center', family='monospace')
            axes[1, 1].set_title('Statistics', fontsize=12, fontweight='bold')
        
        # Intensity histogram
        signal_intensities = image[signal_mask == 1]
        if len(signal_intensities) > 0:
            axes[1, 2].hist(signal_intensities, bins=50, edgecolor='black', alpha=0.7)
            if stats and stats['thresholds']:
                for threshold in stats['thresholds']:
                    axes[1, 2].axvline(threshold, color='red', linestyle='--', 
                                      linewidth=2, label=f'Threshold: {threshold:.1f}')
            axes[1, 2].set_xlabel('Intensity', fontsize=10)
            axes[1, 2].set_ylabel('Frequency', fontsize=10)
            axes[1, 2].set_title('Signal Intensity Distribution', fontsize=12, fontweight='bold')
            axes[1, 2].legend(fontsize=8)
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


class UnifiedSegmentationDensityUNet:
    """
    Unified deep learning model that performs both segmentation and density classification.
    Uses a single U-Net architecture with multi-class output.
    
    Requires TensorFlow and training data.
    
    Example:
        model = UnifiedSegmentationDensityUNet(n_density_classes=3)
        model.build_model()
        # Train model...
        signal_mask, density_class = model.process(image)
    """
    
    def __init__(self, input_shape=(512, 512, 1), n_density_classes=3, learning_rate=1e-4):
        """
        Initialize unified U-Net model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            n_density_classes: Number of density classes (excluding background)
            learning_rate: Learning rate for optimizer
        """
        if not HAS_TF:
            raise ImportError("TensorFlow required for U-Net model. Install with: pip install tensorflow")
        
        self.input_shape = input_shape
        self.n_density_classes = n_density_classes
        self.n_classes = n_density_classes + 1  # +1 for background
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self):
        """Build unified U-Net architecture for segmentation + density classification"""
        inputs = Input(self.input_shape)
        
        # Encoder
        # Block 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        
        # Block 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        
        # Block 3
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
        
        # Block 4
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
        
        # Bottleneck
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        bn5 = BatchNormalization()(conv5)
        
        # Decoder
        # Block 6
        up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn5)
        merge6 = Concatenate(axis=3)([bn4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        bn6 = BatchNormalization()(conv6)
        
        # Block 7
        up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn6)
        merge7 = Concatenate(axis=3)([bn3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        bn7 = BatchNormalization()(conv7)
        
        # Block 8
        up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn7)
        merge8 = Concatenate(axis=3)([bn2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        bn8 = BatchNormalization()(conv8)
        
        # Block 9
        up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn8)
        merge9 = Concatenate(axis=3)([bn1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        bn9 = BatchNormalization()(conv9)
        
        # Output: softmax for multi-class (background + density classes)
        outputs = Conv2D(self.n_classes, 1, activation='softmax')(bn9)
        
        model = Model(inputs=inputs, outputs=outputs, name='UnifiedSegmentationDensityUNet')
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = np.mean(image, axis=2)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.input_shape[1], self.input_shape[0]))
            image = np.array(img_pil).astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=0)  # (1, H, W)
        image = np.expand_dims(image, axis=-1)  # (1, H, W, 1)
        
        return image
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process image: segment and classify densities in one forward pass
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            tuple: (signal_mask, density_classification)
            - signal_mask: Binary mask (1 = signal, 0 = background)
            - density_classification: Multi-class mask (0 = background, 1..n = density classes)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Predict
        pred = self.model.predict(preprocessed, verbose=0)
        
        # Get class predictions (argmax)
        classification = np.argmax(pred[0], axis=-1).astype(np.uint8)
        
        # Separate into signal mask and density classification
        signal_mask = (classification > 0).astype(np.uint8)  # Any non-zero = signal
        
        # Density classification: 0 = background, 1..n = density classes
        density_classification = classification.copy()
        
        return signal_mask, density_classification
    
    def load_model(self, model_path: str):
        """Load trained model from file"""
        if not HAS_TF:
            raise ImportError("TensorFlow required to load model")
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
        # Infer n_classes from output shape
        self.n_classes = self.model.output_shape[-1]
        self.n_density_classes = self.n_classes - 1

    def train(self, train_images, train_masks, val_images=None, val_masks=None,
              batch_size=8, epochs=50, model_save_path=None):
        """
        Train the U-Net on multiple images.

        Args:
            train_images: np.ndarray of shape (N, H, W) or (N, H, W, C)
            train_masks: np.ndarray of shape (N, H, W) with integer labels (0=background, 1..n)
            val_images: optional validation images
            val_masks: optional validation masks
            batch_size: batch size for training
            epochs: number of epochs
            model_save_path: path to save best model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Ensure 4D input: (N, H, W, 1)
        def preprocess_dataset(images):
            if len(images.shape) == 3:  # (N, H, W)
                images = images[..., np.newaxis]
            images = images.astype('float32') / 255.0
            return images
        
        X_train = preprocess_dataset(train_images)
        y_train = train_masks.astype('int32')
        
        if val_images is not None and val_masks is not None:
            X_val = preprocess_dataset(val_images)
            y_val = val_masks.astype('int32')
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        callbacks = []
        if model_save_path:
            checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss',
                                         save_best_only=True, verbose=1)
            callbacks.append(checkpoint)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        callbacks.append(early_stop)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        return history


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED SEGMENTATION & DENSITY CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Example: Create test image
    """test_image = np.ones((512, 512), dtype=np.uint8) * 200  # Bright background
    test_image[200:300, 200:300] = 30   # Very dark (low density)
    test_image[100:150, 400:450] = 50   # Medium dark (medium density)
    test_image[350:400, 100:200] = 70   # Less dark (high density)"""
    
    # Initialize unified model
    print("\nInitializing unified model...")
    model = UnifiedSegmentationDensityModel(
        seg_method='otsu',
        density_method='percentile',
        n_density_classes=3,
        blur_size=5
    )
    
    # Process image
    print("Processing image...")
    signal_mask, density_class, stats = model.process(test_image)
    
    print(f"\nResults:")
    print(f"  Signal mask shape: {signal_mask.shape}")
    print(f"  Density classification shape: {density_class.shape}")
    print(f"  Signal pixels: {np.sum(signal_mask == 1)}")
    print(f"  Density classes found: {np.unique(density_class)}")
    print(f"\nStatistics:")
    print(f"  Thresholds: {stats['thresholds']}")
    for class_id, count in stats['class_counts'].items():
        print(f"  Class {class_id}: {count} pixels")
        if class_id in stats['class_intensities']:
            int_stats = stats['class_intensities'][class_id]
            print(f"    Mean intensity: {int_stats['mean']:.1f} ± {int_stats['std']:.1f}")
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("=" * 60)
    print("  # Process numpy array")
    print("  model = UnifiedSegmentationDensityModel(n_density_classes=3)")
    print("  signal_mask, density_class, stats = model.process(image)")
    print("\n  # Process image file")
    print("  model.process_image('input.tif', 'signal.tif', 'density.tif', visualize=True)")
    print("\n  # Process directory")
    print("  model.process_directory('input_dir/', 'output_dir/')")
    
    if HAS_TF:
        print("\n  # Deep learning model (requires training)")
        print("  unet_model = UnifiedSegmentationDensityUNet(n_density_classes=3)")
        print("  unet_model.build_model()")
        print("  # Train model...")
        print("  signal_mask, density_class = unet_model.process(image)")

