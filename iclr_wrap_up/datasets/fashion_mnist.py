from collections import namedtuple

import numpy as np

from tensorflow import keras
from tensorflow.python.keras import utils as keras_utils


def load():
    """Load the Fashion-MNIST dataset
    Returns: Returns two namedtuples, the first one containing training
        and the second one containing test data respectively. Both come with fields X, y and Y:
        - X is the data
        - y is class, with numbers from 0 to 9
        - Y is class, but coded as a 10-dim vector with one entry set to 1 at the column index corresponding to the class

    """
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
    X_test = np.reshape(X_test, [X_test.shape[0], -1]).astype('float32') / 255.

    X_train = X_train * 2.0 - 1.0
    X_test = X_test * 2.0 - 1.0

    Y_train = keras_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test = keras_utils.to_categorical(y_test, nb_classes).astype('float32')

    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'nb_classes'])
    training = Dataset(X_train, Y_train, y_train, nb_classes)
    test = Dataset(X_test, Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test

    return training, test
