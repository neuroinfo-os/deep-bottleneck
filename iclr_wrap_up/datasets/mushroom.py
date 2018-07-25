import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import namedtuple
from tensorflow import keras
from tensorflow.python.keras import utils as keras_utils


mushrooms = pd.read_csv('datasets/mushroom.csv')


def load():
    n_classes = 2
    y = mushrooms['class=e']
    X = mushrooms.drop(['class=e', 'class=p'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    Y_train = keras_utils.to_categorical(y_train, n_classes).astype('float32')
    Y_test = keras_utils.to_categorical(y_test, n_classes).astype('float32')

    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'n_classes'])
    training = Dataset(X_train, Y_train, y_train, n_classes)
    test = Dataset(X_test, Y_test, y_test, n_classes)

    return training, test
