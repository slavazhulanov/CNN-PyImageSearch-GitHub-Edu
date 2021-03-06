# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
import argparse

import warnings
warnings.filterwarnings('ignore')


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="путь к датасету")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplepreprocessor.SimplePreprocessor(32, 32)
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)


# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with ‘{}‘ penalty".format(r))
    model = SGDClassifier(loss="log_loss", penalty=r, max_iter=10,
    learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r, acc * 100))
