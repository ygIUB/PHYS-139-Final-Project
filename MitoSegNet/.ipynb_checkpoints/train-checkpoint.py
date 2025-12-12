"""
Training script for MitoSegNet on EM data
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from architecture import build_mitosegnet, dice_coefficient

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check GPU availability

print("\nChecking GPU Availability...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    print("TensorFlow will use GPU for training")
    # Enable memory growth to avoid OOM
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error: Could not enable memory growth: {e}")
else:
    print("No GPU found - will use CPU (much slower)")
    print("Training on CPU may take several hours!")



class MitoSegNetTrainer:
    """Trainer class for MitoSegNet model"""

    def __init__(self, img_rows=512, img_cols=512):
        """
        Initialize trainer

        Args:
            img_rows: Image height
            img_cols: Image width
        """
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = None

    def load_data(self, max_samples=6000):
        """
        Load raw training data (no preprocessing yet to save memory)

        Args:
            max_samples: Maximum samples to load

        Returns:
            tuple: (images, masks) as uint8 arrays
        """
        
        print("\nLoading Data...")
        

        # Load raw data (uint8, no preprocessing)
        npy_path = "../npydata"
        print(f"Loading from {npy_path}...")

        # Use memory mapping to avoid loading all into RAM
        imgs_mmap = np.load(npy_path + "/imgs_train.npy", mmap_mode='r')
        masks_mmap = np.load(npy_path + "/imgs_mask_train.npy", mmap_mode='r')

        N = imgs_mmap.shape[0]
        print(f"Found {N} samples in dataset")

        # Subsample BEFORE loading into memory
        if N > max_samples:
            print(f"Selecting random {max_samples} samples...")
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(N, size=max_samples, replace=False)
            idx = np.sort(idx)  # Sort for better disk access

            # Load only the selected samples into memory
            print(f"Loading {max_samples} samples into memory...")
            imgs = imgs_mmap[idx].copy()
            masks = masks_mmap[idx].copy()
            print(f"Loaded {imgs.shape[0]} samples")
        else:
            print(f"Loading all {N} samples into memory...")
            imgs = imgs_mmap[:].copy()
            masks = masks_mmap[:].copy()

        return imgs, masks

    def prepare_data(self, imgs, masks, val_ratio=0.2, seed=42):
        """
        Prepare train/validation split and preprocess

        Args:
            imgs: Training images (uint8)
            masks: Training masks (uint8)
            val_ratio: Validation split ratio
            seed: Random seed

        Returns:
            tuple: (X_train, Y_train, X_val, Y_val)
        """
        
        print("\nPreparing Data...")
        
        print(f"Total samples: {imgs.shape[0]}")

        # Convert to float16 to save memory
        print("Converting to float16 and normalizing...")
        imgs = imgs.astype('float16')
        masks = masks.astype('float16')

        # Normalize images
        imgs /= 255.0
        mean = imgs.mean(axis=0, dtype='float32')
        imgs = imgs - mean.astype('float16')

        # Binarize masks
        masks /= 255.0
        masks[masks > 0.5] = 1.0
        masks[masks <= 0.5] = 0.0

        # Train/validation split
        N = imgs.shape[0]
        val_size = int(N * val_ratio)

        # Use seed=123 to match MoDL_seg implementation
        rng = np.random.default_rng(seed=123)
        indices = rng.permutation(N)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train = imgs[train_idx]
        Y_train = masks[train_idx]
        X_val = imgs[val_idx]
        Y_val = masks[val_idx]

        print(f"train: {X_train.shape[0]}  val: {X_val.shape[0]}")

        return X_train, Y_train, X_val, Y_val

    def build_model(self, learning_rate=1e-4):
        """
        Build MitoSegNet model

        Args:
            learning_rate: Learning rate for optimizer

        Returns:
            Compiled Keras model
        """
        
        print("\nBuilding Model...")
        

        model = build_mitosegnet(
            input_shape=(self.img_rows, self.img_cols, 1),
            learning_rate=learning_rate
        )

        print(f"Model: {model.name}")
        print(f"Total parameters: {model.count_params():,}")

        self.model = model
        return model

    def train(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        batch_size=2,
        epochs=30,
        learning_rate=1e-4,
        output_dir="../models"
    ):
        """
        Train the model

        Args:
            X_train: Training images
            Y_train: Training masks
            X_val: Validation images
            Y_val: Validation masks
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            output_dir: Directory to save models and plots

        Returns:
            Training history
        """
        
        print("\nTraining model...")
        

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build model if not already built
        if self.model is None:
            self.build_model(learning_rate=learning_rate)

        # Create data generators
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_ds = train_ds.shuffle(buffer_size=len(X_train))
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Callbacks
        checkpoint_path = os.path.join(output_dir, "mitosegnet_best.keras")
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train model
        print(f"\nStarting training...")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")

        start_time = datetime.datetime.now()

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=[model_checkpoint, early_stopping, reduce_lr]
        )

        end_time = datetime.datetime.now()
        training_time = end_time - start_time

        print(f"\nTraining complete!")
        print(f"Training time: {training_time}")
        print(f"Best model saved to: {checkpoint_path}")

        # Save training plots
        self.plot_training_history(history, output_dir)

        # Save final model
        final_model_path = os.path.join(output_dir, "mitosegnet_final.keras")
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Save training summary
        self.save_training_summary(history, training_time, output_dir)

        return history

    def plot_training_history(self, history, output_dir):
        """
        Plot and save training curves

        Args:
            history: Training history object
            output_dir: Directory to save plots
        """
        print("\nGenerating training plots...")

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)  # 1-indexed epochs

        # Combined plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy subplot
        ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Loss subplot
        ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {combined_path}")
        plt.close()

        # Individual accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        acc_path = os.path.join(output_dir, 'training_accuracy.png')
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {acc_path}")
        plt.close()

        # Individual loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        loss_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {loss_path}")
        plt.close()

    def save_training_summary(self, history, training_time, output_dir):
        """
        Save training summary to text file

        Args:
            history: Training history
            training_time: Total training time
            output_dir: Directory to save summary
        """
        summary_path = os.path.join(output_dir, 'training_summary.txt')

        with open(summary_path, 'w') as f:
            
            f.write("---MitoSegNet Training Summary---\n")
            

            f.write(f"Training time: {training_time}\n\n")

            f.write("Final Metrics:\n")
            f.write(f"  Training Loss:       {history.history['loss'][-1]:.6f}\n")
            f.write(f"  Training Accuracy:   {history.history['accuracy'][-1]:.6f}\n")
            f.write(f"  Validation Loss:     {history.history['val_loss'][-1]:.6f}\n")
            f.write(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.6f}\n\n")

            f.write("Best Metrics:\n")
            f.write(f"  Best Training Loss:       {min(history.history['loss']):.6f}\n")
            f.write(f"  Best Training Accuracy:   {max(history.history['accuracy']):.6f}\n")
            f.write(f"  Best Validation Loss:     {min(history.history['val_loss']):.6f}\n")
            f.write(f"  Best Validation Accuracy: {max(history.history['val_accuracy']):.6f}\n\n")

            # If Dice coefficient is in history
            if 'dice_coefficient' in history.history:
                f.write("Dice Coefficient:\n")
                f.write(f"  Training:   {history.history['dice_coefficient'][-1]:.6f}\n")
                f.write(f"  Validation: {history.history['val_dice_coefficient'][-1]:.6f}\n")
                f.write(f"  Best Training:   {max(history.history['dice_coefficient']):.6f}\n")
                f.write(f"  Best Validation: {max(history.history['val_dice_coefficient']):.6f}\n\n")

            f.write("=" * 60 + "\n")

        print(f"Training summary saved to: {summary_path}")


def main():
    """Main training function"""
    
    print("\nMitoSegNet Training on EM Data...")
    

    # Configuration
    IMG_SIZE = 512
    MAX_SAMPLES = 6000
    BATCH_SIZE = 2
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    VAL_RATIO = 0.2
    OUTPUT_DIR = "./models"

    print(f"\nConfiguration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Validation ratio: {VAL_RATIO}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Initialize trainer
    trainer = MitoSegNetTrainer(img_rows=IMG_SIZE, img_cols=IMG_SIZE)

    # Load data (with immediate subsampling to avoid OOM)
    imgs, masks = trainer.load_data(max_samples=MAX_SAMPLES)

    # Prepare train/val split
    X_train, Y_train, X_val, Y_val = trainer.prepare_data(
        imgs, masks,
        val_ratio=VAL_RATIO
    )

    # Build model
    trainer.build_model(learning_rate=LEARNING_RATE)

    # Train model
    history = trainer.train(
        X_train, Y_train, X_val, Y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR
    )

   
    print("Training Complete!")
   


if __name__ == "__main__":
    main()