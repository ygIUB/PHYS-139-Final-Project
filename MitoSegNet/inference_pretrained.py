"""
Inference script for PRETRAINED MitoSegNet model
(trained on fluorescent microscopy data)

This demonstrates domain mismatch - the model was trained on fluorescent
images and will likely perform poorly on EM data without fine-tuning.
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tqdm import tqdm


class PretrainedMitoSegNetInference:
    """Inference using pretrained MitoSegNet (fluorescent-trained)"""

    def __init__(self, model_path, threshold=0.5):
        """
        Initialize inference

        Args:
            model_path: Path to pretrained .hdf5 model file
            threshold: Threshold for binary segmentation
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.load_model()

    def load_model(self):
        """Load pretrained MitoSegNet model"""
        print(f"Loading pretrained MitoSegNet from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n\n"
                f"To use pretrained MitoSegNet:\n"
                f"1. Download the pretrained model from the MitoSegNet repository\n"
                f"2. Place it at: {self.model_path}\n\n"
                f"Note: This model was trained on fluorescent microscopy data,\n"
                f"NOT EM data, so performance on EM images will be poor."
            )

        try:
            self.model = load_model(self.model_path, compile=False)
            print(f"Model loaded successfully")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess image for inference

        Args:
            image_path: Path to input image
            target_size: Target size (height, width), defaults to model input

        Returns:
            tuple: (preprocessed_image, original_size)
        """
        # Load image as grayscale
        img = load_img(image_path, color_mode='grayscale')
        original_size = img.size  # (width, height)

        # Get model input size
        if target_size is None:
            input_shape = self.model.input_shape
            target_size = (input_shape[1], input_shape[2])  # (height, width)

        # Resize if needed
        if img.size != (target_size[1], target_size[0]):
            img = img.resize((target_size[1], target_size[0]))

        # Convert to array and normalize
        img_array = img_to_array(img)
        img_array = img_array.astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_size

    def postprocess_mask(self, pred_mask, original_size):
        """
        Postprocess prediction to binary mask

        Args:
            pred_mask: Predicted mask from model
            original_size: Original image size (width, height)

        Returns:
            Binary mask as uint8 array
        """
        # Remove batch dimension
        if len(pred_mask.shape) == 4:
            pred_mask = pred_mask[0, :, :, 0]

        # Resize back to original size
        pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        pred_img = pred_img.resize(original_size, Image.BILINEAR)

        # Threshold to binary
        pred_array = np.array(pred_img).astype('float32') / 255.0
        binary_mask = (pred_array > self.threshold).astype(np.uint8) * 255

        return binary_mask

    def predict(self, image_path):
        """
        Run inference on a single image

        Args:
            image_path: Path to input image

        Returns:
            Binary segmentation mask (0-255)
        """
        # Preprocess
        img_array, original_size = self.preprocess_image(image_path)

        # Predict
        pred_mask = self.model.predict(img_array, verbose=0)

        # Postprocess
        binary_mask = self.postprocess_mask(pred_mask, original_size)

        return binary_mask

    def predict_batch(self, image_dir, output_dir, img_extension='tif'):
        """
        Run inference on a directory of images

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save predictions
            img_extension: Image file extension

        Returns:
            Number of images processed
        """
        print(f"\nRunning batch inference...")
        print(f"  Input dir: {image_dir}")
        print(f"  Output dir: {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get all images
        import glob
        image_paths = glob.glob(os.path.join(image_dir, f"*.{img_extension}"))
        print(f"  Found {len(image_paths)} images")

        if len(image_paths) == 0:
            print(f"No images found with extension .{img_extension}")
            return 0

        # Process each image
        for image_path in tqdm(image_paths, desc="Processing"):
            # Get filename
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)

            # Run inference
            mask = self.predict(image_path)

            # Save mask
            Image.fromarray(mask).save(output_path)

        print(f"Saved {len(image_paths)} predictions to {output_dir}")
        return len(image_paths)

    def visualize_prediction(self, image_path, output_path=None):
        """
        Visualize prediction alongside original image

        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
        """
        # Load original image
        original = np.array(Image.open(image_path))

        # Get prediction
        prediction = self.predict(image_path)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image (EM)', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Predicted Mask\n(Pretrained on Fluorescent)', fontsize=12)
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(original, cmap='gray')
        axes[2].imshow(prediction, cmap='Reds', alpha=0.5)
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main inference function"""
    print("\n" + "=" * 60)
    print("PRETRAINED MITOSEGNET INFERENCE (FLUORESCENT MODEL)")
    print("=" * 60)
    print("\nWARNING: This model was trained on fluorescent microscopy,")
    print("NOT electron microscopy. Performance will likely be poor.")
    print("This demonstrates the need for domain adaptation.\n")

    # Configuration
    MODEL_PATH = "./models/MitoSegNet_pretrained.hdf5"
    TEST_IMAGE_DIR = "../test/images"
    OUTPUT_DIR = "./outputs/pretrained_predictions"
    THRESHOLD = 0.5

    print(f"Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Test images: {TEST_IMAGE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Threshold: {THRESHOLD}")

    # Initialize inference
    try:
        inference = PretrainedMitoSegNetInference(
            model_path=MODEL_PATH,
            threshold=THRESHOLD
        )
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nSkipping pretrained inference - model file not available.")
        return

    # Run batch inference
    if os.path.exists(TEST_IMAGE_DIR):
        n_processed = inference.predict_batch(
            image_dir=TEST_IMAGE_DIR,
            output_dir=OUTPUT_DIR
        )
        print(f"\nProcessed {n_processed} images")
    else:
        print(f"\nTest image directory not found: {TEST_IMAGE_DIR}")
        print("Please provide test images to run inference.")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()