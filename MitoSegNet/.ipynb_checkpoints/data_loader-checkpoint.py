"""
Data loader for MitoSegNet training and evaluation
"""

import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


class DataLoader(object):
    """
    Data loader for mitochondria segmentation
    Loads images and masks, preprocesses them, and saves as .npy files
    """

    def __init__(
        self,
        out_rows=512,
        out_cols=512,
        data_path="../deform/train",
        label_path="../deform/label",
        npy_path="../npydata",
        img_type="tif"
    ):
        """
        Initialize data loader

        Args:
            out_rows: Output image height
            out_cols: Output image width
            data_path: Path to training images
            label_path: Path to label masks
            npy_path: Path to save .npy files
            img_type: Image file extension
        """
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.npy_path = npy_path

    def create_train_data(self):
        """
        Load training images and masks, convert to numpy arrays
        Saves as imgs_train.npy and imgs_mask_train.npy
        """
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        # Get all image files
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(f"Found {len(imgs)} images")

        # Initialize arrays
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        # Load each image and label
        for i, imgname in enumerate(imgs):
            midname = os.path.basename(imgname.replace('\\', '/'))

            # Load image and label as grayscale
            img = load_img(self.data_path + "/" + midname, color_mode='grayscale')
            label = load_img(self.label_path + "/" + midname, color_mode='grayscale')

            # Convert to arrays
            img = img_to_array(img)
            label = img_to_array(label)

            imgdatas[i] = img
            imglabels[i] = label

            if i % 100 == 0:
                print(f'Done: {i}/{len(imgs)} images')

        print('Loading done')

        # Save as .npy files
        os.makedirs(self.npy_path, exist_ok=True)
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

        print(f'Saved to {self.npy_path}')
        print(f'  - imgs_train.npy: {imgdatas.shape}')
        print(f'  - imgs_mask_train.npy: {imglabels.shape}')

    def load_train_data(self):
        """
        Load preprocessed training data from .npy files
        Normalizes images and binarizes masks

        Returns:
            tuple: (normalized_images, binary_masks)
        """
        print("Loading training data from .npy files...")

        # Load arrays
        train = np.load(self.npy_path + "/imgs_train.npy")
        mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")

        print(f"Loaded {train.shape[0]} samples")

        # Convert to float32 and normalize
        train = train.astype('float32')
        mask_train = mask_train.astype('float32')

        # Normalize images to [0, 1]
        train /= 255.0

        # Mean subtraction
        mean = train.mean(axis=0)
        train -= mean

        # Normalize and binarize masks
        mask_train /= 255.0
        mask_train[mask_train > 0.5] = 1
        mask_train[mask_train <= 0.5] = 0

        print("Preprocessing complete")
        print(f"  Images: min={train.min():.3f}, max={train.max():.3f}")
        print(f"  Masks: unique values={np.unique(mask_train)}")

        return train, mask_train


if __name__ == "__main__":
    # Example usage
    print("MitoSegNet Data Loader")
    print("=" * 50)

    # Create data loader
    loader = DataLoader(
        out_rows=512,
        out_cols=512,
        data_path="../deform/train",
        label_path="../deform/label",
        npy_path="../npydata"
    )

    # Option 1: Create .npy files from images
    # loader.create_train_data()

    # Option 2: Load existing .npy files
    try:
        imgs, masks = loader.load_train_data()
        print(f"\nLoaded data shapes:")
        print(f"  Images: {imgs.shape}")
        print(f"  Masks: {masks.shape}")
    except FileNotFoundError:
        print("\nNo .npy files found. Run create_train_data() first.")