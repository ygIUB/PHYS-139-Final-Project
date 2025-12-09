"""
Inference script for MitoSegNet trained on EM data
Uses the model we trained from scratch on electron microscopy images
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


class TrainedMitoSegNetInference:
    """Inference using MitoSegNet trained on EM data"""

    def __init__(self, model_path="./models/mitosegnet_best.keras", threshold=0.5):
        """
        Initialize inference

        Args:
            model_path: Path to trained model file
            threshold: Threshold for binary segmentation
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.load_model()

    def load_model(self):
        """Load trained MitoSegNet model"""
        print(f"Loading trained MitoSegNet from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n\n"
                f"Please train the model first by running: python train.py"
            )

        try:
            self.model = load_model(self.model_path, compile=False)
            print(f"Model loaded successfully")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        Follows the same preprocessing as training

        Args:
            image_path: Path to input image

        Returns:
            tuple: (preprocessed_image, original_image_array)
        """
        # Load image as grayscale
        img = load_img(image_path, color_mode='grayscale')
        img_array = img_to_array(img)

        # Store original for later
        original = img_array.copy()

        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0

        # Mean subtraction (using global mean approximation)
        # In practice, you might want to use the exact mean from training
        mean = img_array.mean()
        img_array -= mean

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original

    def predict(self, image_path):
        """
        Run inference on a single image

        Args:
            image_path: Path to input image

        Returns:
            tuple: (binary_mask, probability_mask, original_image)
        """
        # Preprocess
        img_array, original = self.preprocess_image(image_path)

        # Predict
        prob_mask = self.model.predict(img_array, verbose=0)

        # Remove batch and channel dimensions
        prob_mask = prob_mask[0, :, :, 0]

        # Threshold to binary
        binary_mask = (prob_mask > self.threshold).astype(np.uint8) * 255

        # Convert probability to 0-255 range
        prob_mask_viz = (prob_mask * 255).astype(np.uint8)

        return binary_mask, prob_mask_viz, original.astype(np.uint8)

    def predict_batch(self, image_dir, output_dir, img_extension='tif', save_probability=False):
        """
        Run inference on a directory of images

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save predictions
            img_extension: Image file extension
            save_probability: If True, also save probability maps

        Returns:
            Number of images processed
        """
        print(f"\nRunning batch inference...")
        print(f"  Input dir: {image_dir}")
        print(f"  Output dir: {output_dir}")

        # Create output directories
        binary_dir = os.path.join(output_dir, "binary")
        os.makedirs(binary_dir, exist_ok=True)

        if save_probability:
            prob_dir = os.path.join(output_dir, "probability")
            os.makedirs(prob_dir, exist_ok=True)

        # Get all images
        image_paths = sorted(glob.glob(os.path.join(image_dir, f"*.{img_extension}")))
        print(f"  Found {len(image_paths)} images")

        if len(image_paths) == 0:
            print(f"No images found with extension .{img_extension}")
            return 0

        # Process each image
        for image_path in tqdm(image_paths, desc="Processing"):
            # Get filename
            filename = os.path.basename(image_path)

            # Run inference
            binary_mask, prob_mask, _ = self.predict(image_path)

            # Save binary mask
            binary_path = os.path.join(binary_dir, filename)
            Image.fromarray(binary_mask).save(binary_path)

            # Save probability map if requested
            if save_probability:
                prob_path = os.path.join(prob_dir, filename)
                Image.fromarray(prob_mask).save(prob_path)

        print(f"\nSaved {len(image_paths)} predictions to {output_dir}")
        return len(image_paths)

    def visualize_prediction(self, image_path, output_path=None, show_probability=True):
        """
        Visualize prediction alongside original image

        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            show_probability: If True, show probability map
        """
        # Get prediction
        binary_mask, prob_mask, original = self.predict(image_path)

        # Create visualization
        n_cols = 4 if show_probability else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))

        # Original image
        axes[0].imshow(original[:, :, 0], cmap='gray')
        axes[0].set_title('Original Image (EM)', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Binary prediction
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title(f'Binary Prediction\n(threshold={self.threshold})', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(original[:, :, 0], cmap='gray')
        axes[2].imshow(binary_mask, cmap='Reds', alpha=0.4)
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Probability map
        if show_probability:
            im = axes[3].imshow(prob_mask, cmap='viridis')
            axes[3].set_title('Probability Map', fontsize=12, fontweight='bold')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_multiple(self, image_dir, n_samples=6, output_path=None):
        """
        Visualize multiple predictions in a grid

        Args:
            image_dir: Directory containing images
            n_samples: Number of samples to visualize
            output_path: Path to save visualization (optional)
        """
        # Get random images
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

        if len(image_paths) == 0:
            print(f"No images found in {image_dir}")
            return

        # Sample random images
        n_samples = min(n_samples, len(image_paths))
        indices = np.random.choice(len(image_paths), n_samples, replace=False)
        sampled_paths = [image_paths[i] for i in indices]

        # Create grid
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, image_path in enumerate(sampled_paths):
            binary_mask, _, original = self.predict(image_path)
            filename = os.path.basename(image_path)

            # Original
            axes[idx, 0].imshow(original[:, :, 0], cmap='gray')
            axes[idx, 0].set_title(f'{filename}', fontsize=10)
            axes[idx, 0].axis('off')

            # Prediction
            axes[idx, 1].imshow(binary_mask, cmap='gray')
            axes[idx, 1].set_title('Prediction', fontsize=10)
            axes[idx, 1].axis('off')

            # Overlay
            axes[idx, 2].imshow(original[:, :, 0], cmap='gray')
            axes[idx, 2].imshow(binary_mask, cmap='Reds', alpha=0.4)
            axes[idx, 2].set_title('Overlay', fontsize=10)
            axes[idx, 2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main inference function"""
    print("\n" + "=" * 60)
    print("MITOSEGNET INFERENCE (TRAINED ON EM DATA)")
    print("=" * 60)

    # Configuration
    MODEL_PATH = "./models/mitosegnet_best.keras"
    TEST_IMAGE_DIR = "../test/images"
    OUTPUT_DIR = "./outputs/trained_predictions"
    THRESHOLD = 0.5

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Test images: {TEST_IMAGE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Threshold: {THRESHOLD}")

    # Initialize inference
    try:
        inference = TrainedMitoSegNetInference(
            model_path=MODEL_PATH,
            threshold=THRESHOLD
        )
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    # Run batch inference
    if os.path.exists(TEST_IMAGE_DIR):
        n_processed = inference.predict_batch(
            image_dir=TEST_IMAGE_DIR,
            output_dir=OUTPUT_DIR,
            save_probability=True
        )
        print(f"\nProcessed {n_processed} images")

        # Create sample visualizations
        viz_path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
        inference.visualize_multiple(
            image_dir=TEST_IMAGE_DIR,
            n_samples=6,
            output_path=viz_path
        )
    else:
        print(f"\nTest image directory not found: {TEST_IMAGE_DIR}")
        print("Please provide test images to run inference.")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()