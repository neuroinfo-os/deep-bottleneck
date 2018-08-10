import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import namedtuple
from tensorflow import keras
from tensorflow.python.keras import utils as keras_utils


def load():
    """Load the mushroom dataset
    Mushrooms are to be classified as either edible or poisonous
    Returns:
        Returns two namedtuples, the first one containing training
        and the second one containing test data respectively. Both come with fields X, y and Y:
        - X is the data
        - y is class, with numbers from 0 to 1
        - Y is class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
    """
    mushrooms = pd.read_csv('datasets/mushroom.csv')

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
