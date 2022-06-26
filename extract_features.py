# import the necessary packages
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.utils import img_to_array
from keras.utils import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os


# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
ap.add_argument("-o", "--output", required = True, help = "path to output HDF5 file")
ap.add_argument("-b", "--batch_size", type = int, default = 32,
    help = "batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type = int, default = 1000,
    help = "size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args["batch_size"]

# grab the list of images that we will be describing then randomly shuffle them
# to allow for easy training and testing splits via array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load VGG16 network
print("[INFO] loading networks...")
model = VGG16(weights = "imagenet", include_top = False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args["output"],
    dataKey = "features", bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), widgets = widgets).start()

# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the list
    # of actual images that will be passed through the network for feature extraction
    batchPaths = imagePaths[i : i + bs]
    batchLabels = labels[i : i + bs]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224 x 224 pixels
        image = load_img(imagePath, target_size = (224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and (2) subtracting
        # the mean RGB pixel intensity from the ImageNet dataset
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)

    # reshape the features so that each image is represented by a flattened
    # feature vector of the 'MaxPooling2D' outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add features and labels to our HDF dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close dataset
dataset.close()
pbar.finish()


"python extract_features.py --dataset pyimagesearch/datasets/animals/images --output pyimagesearch/datasets/animals/hdf5/features.hdf5"
"python extract_features.py --dataset pyimagesearch/datasets/flower17/images --output pyimagesearch/datasets/flower17/hdf5/features.hdf5"


