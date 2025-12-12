import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Dropout, BatchNormalization, concatenate,
    Conv2DTranspose, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from data_load import *
from ourResUnet import build_our_resunet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class myUnet(object):
   def __init__(self, img_rows = 512, img_cols = 512, model_type="baseline"):
      self.img_rows = img_rows
      self.img_cols = img_cols
      self.model_type = model_type   # "baseline" or "our"

   def load_data(self):
      mydata = DataProcess(self.img_rows, self.img_cols)
      imgs_train, imgs_mask_train = mydata.load_train_data()
      return imgs_train, imgs_mask_train

   def get_our_resunet(self):
      return build_our_resunet(input_shape=(self.img_rows, self.img_cols, 1))

   def get_unet(self):
      inputs = Input((self.img_rows, self.img_cols,1))

      conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
      print("conv1 shape:", conv1.shape)
      conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
      print ("conv1 shape:",conv1.shape)
      pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      print ("pool1 shape:",pool1.shape)


      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
      print("conv2 shape:", conv2.shape)
      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
      print ("conv2 shape:",conv2.shape)
      pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
      print ("pool2 shape:",pool2.shape)


      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
      print("conv3 shape:", conv3.shape)
      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
      print ("conv3 shape:",conv3.shape)
      pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
      print ("pool3 shape:",pool3.shape)


      conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
      print("conv4 shape:", conv4.shape)
      conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
      print("conv4 shape:", conv4.shape)
      drop4 = Dropout(0.5)(conv4)
      pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
      print("pool4 shape:", pool4.shape)

      conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
      print("conv5 shape:", conv5.shape)
      conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
      print("conv5 shape:", conv5.shape)
      drop5 = Dropout(0.5)(conv5)


      up6 = Conv2DTranspose(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
      print("up6 shape:", up6.shape)
      merge6 = Concatenate(axis=3)([drop4, up6])
      print("merge6 shape:", merge6.shape)
      conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
      print("conv6 shape:", conv6.shape)
      conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
      print("conv6 shape:", conv6.shape)

      up7 = Conv2DTranspose(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
      print("up7 shape:", up7.shape)
      merge7 = Concatenate(axis=3)([conv3, up7])
      print("merge7 shape:", merge7.shape)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
      print("conv7 shape:", conv7.shape)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
      print("conv7 shape:", conv7.shape)


      up8 = Conv2DTranspose(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
      print("up8 shape:", up8.shape)
      merge8 = Concatenate(axis=3)([conv2, up8])
      print("merge8 shape:", merge8.shape)
      conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
      print("conv8 shape:", conv8.shape)
      conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
      print("conv8 shape:", conv8.shape)


      up9 = Conv2DTranspose(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
      print("up9 shape:", up9.shape)
      merge9 = Concatenate(axis=3)([conv1, up9])
      print("merge9 shape:", merge9.shape)
      conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
      print("conv9 shape:", conv9.shape)
      conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
      print("conv9 shape:", conv9.shape)
      conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
      print("conv9 shape:", conv9.shape)
      conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
      print("conv10 shape:", conv10.shape)

      model = Model(inputs = inputs, outputs = conv10)
      model.compile(optimizer = Adam(learning_rate=1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
      return model

 

   def train(self):
         print("loading data")

         imgs = np.load("../npydata/imgs_train.npy").astype("float32")
         masks = np.load("../npydata/imgs_mask_train.npy").astype("float32")
         print("total samples (full):", imgs.shape[0])

         MAX_SAMPLES = 6000 #select 6000
         N = imgs.shape[0]
         if N > MAX_SAMPLES:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(N, size=MAX_SAMPLES, replace=False)
            imgs = imgs[idx]
            masks = masks[idx]
            print(f"subsampled to {MAX_SAMPLES} samples for GPU training")
         else:
            print("use all samples for GPU training")

         imgs = imgs.astype("float16")
         masks = masks.astype("float16")

         imgs /= 255.0
         mean = imgs.mean(axis=0, dtype="float32")
         imgs = imgs - mean.astype("float16")

         masks /= 255.0
         masks[masks > 0.5] = 1.0
         masks[masks <= 0.5] = 0.0

         print("after subsample:", imgs.shape[0])

        #train/validation split（0.8 / 0.2)
         N = imgs.shape[0]
         val_ratio = 0.2
         val_size = int(N * val_ratio)

         rng = np.random.default_rng(seed=123)
         indices = rng.permutation(N)

         val_idx = indices[:val_size]
         train_idx = indices[val_size:]

         X_train = imgs[train_idx]
         Y_train = masks[train_idx]
         X_val   = imgs[val_idx]
         Y_val   = masks[val_idx]

         print(f"train: {X_train.shape[0]}  val: {X_val.shape[0]}")

         BATCH_SIZE   = 2
         TOTAL_EPOCHS = 30

         train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
         train_ds = train_ds.shuffle(buffer_size=len(X_train))
         train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

         val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
         val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


         print("Building a fresh model...")
         if self.model_type == "baseline":
            print("Using baseline U-Net")
            model = self.get_unet()
         elif self.model_type == "our":
            print("Using OurResUNet")
            model = self.get_our_resunet()
         else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
         print("got model:", model.name)


         model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
         )

         CHECKPOINT_PATH = "../model/U-RNet+_gpu_10ep.keras"
         model_checkpoint = ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
         )

         starttrain = datetime.datetime.now()
         print("Fitting model...")

         history = model.fit(
            train_ds,
            epochs=TOTAL_EPOCHS,
            verbose=1,
            validation_data=val_ds,
            callbacks=[model_checkpoint],
         )

         endtrain = datetime.datetime.now()
         print("train time: %s seconds" % (endtrain - starttrain))


         acc      = history.history["accuracy"]
         val_acc  = history.history["val_accuracy"]
         loss     = history.history["loss"]
         val_loss = history.history["val_loss"]
         epochs   = range(len(acc))

         plt.figure()
         plt.plot(epochs, acc, "b", label="training accuracy")
         plt.plot(epochs, val_acc, ":r", label="validation accuracy")
         plt.title("Accuracy")
         plt.xlabel("Epoch")
         plt.ylabel("Accuracy")
         plt.legend()
         plt.savefig("../model/Accuracy.png")

         plt.figure()
         plt.plot(epochs, loss, "b", label="training loss")
         plt.plot(epochs, val_loss, ":r", label="validation loss")
         plt.title("Loss")
         plt.xlabel("Epoch")
         plt.ylabel("Loss")
         plt.legend()
         plt.savefig("../model/Loss.png")

         plt.show()

         with open("../model/unet.txt", "wt") as ft:
            ft.write("loss: %.6s\n" % (loss[-1]))
            ft.write("accuracy: %.6s\n" % (acc[-1]))


if __name__ == '__main__':

   myunet = myUnet(model_type="baseline") # please choose between "baseline" or "our"
   myunet.train()