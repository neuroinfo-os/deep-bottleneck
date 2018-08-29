import numpy as np

from tensorflow import keras

from deep_bottleneck.datasets.base_dataset import Dataset


def load():
    """Load the Fashion-MNIST dataset

    Returns:
        The fashion mnist dataset.
    """
    n_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.0
    X_test = np.reshape(X_test, [X_test.shape[0], -1]).astype('float32') / 255.0

    X_train = X_train * 2.0 - 1.0
    X_test = X_test * 2.0 - 1.0

    dataset = Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)

    return dataset
