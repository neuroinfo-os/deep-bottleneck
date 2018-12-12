from collections import namedtuple

import numpy as np

from tensorflow import keras
from tensorflow.python.keras import utils as keras_utils
from deep_bottleneck.datasets.base_dataset import Dataset


def load():
    """Load the CIFAR 10 dataset
    Returns:
        CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes
        Returns two namedtuples, the first one containing training
        and the second one containing test data respectively. Both come with fields X, y and Y:
        - X is the data
        - y is class, with numbers from 0 to 9
        - Y is class, but coded as a 10-dim vector with one entry set to 1 at the column index corresponding to the class
    """
    n_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.0
    X_test = np.reshape(X_test, [X_test.shape[0], -1]).astype('float32') / 255.0

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    X_train = X_train * 2.0 - 1.0
    X_test = X_test * 2.0 - 1.0

    dataset = Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)
    
    return dataset
