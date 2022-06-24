'''python keras_mnist.py --output output/keras_mnist.png'''
'''импорт необходимых библиотек'''
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential # прямая связь нейронов
from keras.layers.core import Dense # полносвязные слои
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="путь к выходному графику потерь\точности")
args = vars(ap.parse_args())

'''возьмите набор данных MNIST (если вы запускаете это впервые
скрипт, загрузка может занять минуту — набор данных MNIST объемом 55МБ
будет скачиваться)'''
print("[INFO] загрузка датасета MNIST...")
dataset = datasets.fetch_openml("mnist_784")

'''масштабируйте необработанные интенсивности пикселей в диапазоне [0, 1.0], затем
построить тренировочный и тестовый сплиты'''
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

'''преобразовать метки из целых чисел в векторы'''
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

'''определить архитектуру 784-256-128-10 с помощью Keras'''
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))

'''обучить модель с помощью SGD'''
print("[INFO] тренеровка сети...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

'''оценить сеть'''
print("[INFO] оценка сети...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

'''построить график потерь и точности'''
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="потери при обучении")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="потери при проверке")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="точность при обучении")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="точность при проверке")
plt.title("Потери при обучении и точность")
plt.xlabel("Эпохи")
plt.ylabel("Потери/точность")
plt.legend()
plt.savefig(args["output"])
