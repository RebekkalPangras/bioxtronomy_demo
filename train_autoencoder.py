# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from pymagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


def visualize_predictions(decoded, gt, samples=10):
    # initialize our list of output images
    outputs = None

    # loop over our number of output samples
    for i in range(0, samples):
        # grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # return the output images
    return outputs


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained autoencoder")
ap.add_argument("-v", "--vis", type=str, default="recon_vis.png",
                help="path to output reconstruction visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 2

# load the MNIST dataset
print("[INFO] loading dataset...")

import numpy as np
import os
import PIL
import pathlib
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
# from keras.preprocessing.image import ImageDataGenerator
#
# datagen = ImageDataGenerator(rescale=1./255)
# # datagen = ImageDataGenerator()
#
# batch_size = 2
# img_height = 180
# img_width = 180
#
# ...
# # load and iterate training dataset
# trainX = datagen.flow_from_directory('/content/bioxtronomy/data/train',
#                                      # batch_size=2,
#                                      class_mode=None
#                                      )
# # load and iterate validation dataset
# val_it = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
# load and iterate test dataset
# testX = datagen.flow_from_directory('/content/bioxtronomy/data/test',
#                                      # batch_size=2,
#                                      class_mode=None)
# data_dir = '/content/bioxtronomy/data'
#
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width))
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width))
#
# trainX = tf.data.Dataset.from_tensor_slices(train_ds)
# testX = tf.data.Dataset.from_tensor_slices(val_ds)
# trainX = train_ds
# testX = val_ds
# ((trainX, _), (testX, _)) = mnist.load_data()

import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# PATH = os.getcwd()

train_path = '/content/bioxtronomy/data/train/bio/'
train_batch = os.listdir(train_path)
x_train = []

# if data are in form of images
for sample in train_batch:
    img_path = train_path + sample
    x = image.load_img(img_path, color_mode="grayscale", target_size=(64,64))
    img_array = img_to_array(x)
    # preprocessing if required
    x_train.append(img_array)

test_path = '/content/bioxtronomy/data/test/astronomy/'
test_batch = os.listdir(test_path)
x_test = []

for sample in test_batch:
    img_path = test_path + sample
    x = image.load_img(img_path, color_mode="grayscale", target_size=(180,180))
    img_array = img_to_array(x)
    # preprocessing if required
    x_test.append(img_array)

# finally converting list into numpy array
trainX = np.array(x_train)
testX = np.array(x_test)

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = ConvAutoencoder.build(64, 64, 1)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

print("Started Training!")

# train the convolutional autoencoder
H = autoencoder.fit(
    trainX, trainX,
    validation_data=(testX, testX),
    epochs=EPOCHS,
    batch_size=BS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite(args["vis"], vis)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save(args["model"], save_format="h5")
