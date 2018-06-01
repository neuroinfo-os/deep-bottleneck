import numpy as np
from collections import namedtuple


def construct_full_dataset(training, test):
    """
    :param training: Namedtuple with fields X, y and Y:
    X is the training data
    y is training class, with numbers from 0 to 1
    Y is training class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
    :param test: Namedtuple with fields X, y and Y:
    X is the test data
    y is test class, with numbers from 0 to 1
    Y is test class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
    :return: A new Namedtuple with fields X, y and Y containing the concatenation of training and test data
    """
    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    X = np.concatenate((training.X, test.X))
    y = np.concatenate((training.y, test.y))
    Y = np.concatenate((training.Y, test.Y))
    return Dataset(X, Y, y, training.nb_classes)


def shuffle_in_unison_inplace(a, b):
    """ Shullfes both array a and b randomly in unison ""
    :param a: An Array, containing data samples
    :param b: An Array, containing labels respective to a
    :return: Both arrays shuffled in the same way
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """ Divided the data to train and test and shuffle it """
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train, data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets


def is_dense_like(layer):
    """ Check whether a layer has attribute 'kernel', which is true for dense-like layers
    :param layer: Keras layer to check for attribute 'kernel'
    :return: True if layer has attribute 'kernel', False otherwise
    """
    return hasattr(layer, 'kernel')