# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import argparse
import pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained autoencoder")
ap.add_argument("-i", "--index", type=str, required=True,
	help="path to output features index file")
args = vars(ap.parse_args())

# load the MNIST dataset
print("[INFO] loading training split...")
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
    x = image.load_img(img_path, color_mode="grayscale", target_size=(64,64))
    img_array = img_to_array(x)
    # preprocessing if required
    x_test.append(img_array)

# finally converting list into numpy array
trainX = np.array(x_train)
testX = np.array(x_test)

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0


# add a channel dimension to every image in the training split, then
# scale the pixel intensities to the range [0, 1]
# trainX = np.expand_dims(trainX, axis=-1)
# trainX = trainX.astype("float32") / 255.0

# load our autoencoder from disk
print("[INFO] loading autoencoder model...")
autoencoder = load_model(args["model"])

# create the encoder model which consists of *just* the encoder
# portion of the autoencoder
encoder = Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)

# quantify the contents of our input images using the encoder
print("[INFO] encoding images...")
features = encoder.predict(trainX)

# construct a dictionary that maps the index of the MNIST training
# image to its corresponding latent-space representation
indexes = list(range(0, trainX.shape[0]))
data = {"indexes": indexes, "features": features}

# write the data dictionary to disk
print("[INFO] saving index...")
f = open(args["index"], "wb")
f.write(pickle.dumps(data))
f.close()