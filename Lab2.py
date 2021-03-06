import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import math
import numpy
import random
import sys


random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = str(sys.argv[1])
DATASET = str(sys.argv[2])

# DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    NIH = 28
    NIW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    NIH = 28
    NIW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    NIH = 28
    NIW = 28
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    NIH = 28
    NIW = 28
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    NIH = 28
    NIW = 28
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072

IF_CROP = True


def cifar_preprocess(images, outputs):
    if IF_CROP:
        A = np.zeros((images.shape[0] * 3, NIH, NIW, IZ))
        B = np.zeros((outputs.shape[0] * 3, NUM_CLASSES))

        count = 0
        for i in range(0, A.shape[0], 3):
            A[i] = tf.image.random_crop(
                images[count], size=[NIH, NIW, IZ])
            B[i] = outputs[count]

            A[i+1] = tf.image.random_crop(
                images[count], size=[NIH, NIW, IZ])
            B[i+1] = outputs[count]

            A[i+2] = tf.image.random_crop(
                images[count], size=[NIH, NIW, IZ])
            B[i+2] = outputs[count]

            count += 1

        return A, B
    else:
        return images, outputs


# =========================<Classifier Functions>================================


def guesserClassifier(xTest):
    ans = []
    for _ in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps=6):
    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))
    lt = keras.losses.categorical_crossentropy
    model.compile(optimizer='adam',
                  loss=lt, metrics=['accuracy'])
    model.fit(x, y, batch_size=30, epochs=eps)
    return model


def buildTFConvNet(x, y, eps=10, dropout=True, dropRate=0.2):
    if IF_CROP:
        input_shape = (NIH, NIW, IZ)
    else:
        input_shape = (IH, IW, IZ)

    model = keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=input_shape, padding="valid"))

    if DATASET != "mnist_d":
        model.add(tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding="valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    # if DATASET != "mnist_f" and DATASET != "cifar_10":
    model.add(tf.keras.layers.BatchNormalization())
    if dropout:
        model.add(tf.keras.layers.Dropout(dropRate))

    model.add(tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding="valid"))
    if DATASET != "mnist_d":
        model.add(tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding="valid"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    # if DATASET != "mnist_f" and DATASET != "cifar_10":
    model.add(tf.keras.layers.BatchNormalization())
    if dropout:
        model.add(tf.keras.layers.Dropout(dropRate))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    # if DATASET != "mnist_f" and DATASET != "cifar_10":
    model.add(tf.keras.layers.BatchNormalization())
    if dropout:
        model.add(tf.keras.layers.Dropout(dropRate))

    if DATASET != "mnist_d":
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        # if DATASET != "mnist_f" and DATASET != "cifar_10":
        model.add(tf.keras.layers.BatchNormalization())
        if dropout:
            model.add(tf.keras.layers.Dropout(dropRate))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # if DATASET != "mnist_f" and DATASET != "cifar_10":
    model.add(tf.keras.layers.BatchNormalization())
    if dropout:
        model.add(tf.keras.layers.Dropout(dropRate))

    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    lt = keras.losses.categorical_crossentropy
    model.compile(optimizer='adam',
                  loss=lt, metrics=['accuracy'])
    if IF_CROP:
        model.fit(x, y, batch_size=90, epochs=eps)
    else:

        model.fit(x, y, batch_size=30, epochs=eps)

    return model

# =========================<Pipeline Functions>==================================


def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
        yTrainP = to_categorical(yTrain, NUM_CLASSES)
        yTestP = to_categorical(yTest, NUM_CLASSES)

    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
        yTrainP = to_categorical(yTrain, NUM_CLASSES)
        yTestP = to_categorical(yTest, NUM_CLASSES)
        if DATASET != "mnist_d" and DATASET != "mnist_f":
            xTrainP, yTrainP = cifar_preprocess(xTrainP, yTrainP)
            xTestP, yTestP = cifar_preprocess(xTestP, yTestP)

    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")

        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    _, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):
            acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
