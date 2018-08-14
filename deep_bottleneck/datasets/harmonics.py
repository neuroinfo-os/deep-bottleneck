import scipy.io as sio
from pathlib2 import Path
from collections import namedtuple

import numpy as np

from tensorflow.python.keras import utils as keras_utils
from deep_bottleneck import utils


def load(nb_dir = ''):
    """ Load the Information Bottleneck harmonics dataset
    Returns:
        Returns two namedtuples, the first one containing training
        and the second one containing test data respectively. Both come with fields X, y and Y:
        - X is the data
        - y is class, with numbers from 0 to 1
        - Y is class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
    """
    ID = '2017_12_21_16_51_3_275766'
    n_classes = 2
    data_file = Path(nb_dir + 'datasets/IB_data_' + str(ID) + '.npz')
    if data_file.is_file():
        data = np.load(nb_dir + 'datasets/IB_data_' + str(ID) + '.npz')
    else:
        import_IB_data_from_mat(ID, nb_dir)
        data = np.load(nb_dir + 'datasets/IB_data_' + str(ID) + '.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']

    Y_train = keras_utils.to_categorical(y_train, n_classes).astype('float32')
    Y_test = keras_utils.to_categorical(y_test, n_classes).astype('float32')

    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'n_classes'])
    training = Dataset(X_train, Y_train, y_train, int(n_classes))
    test = Dataset(X_test, Y_test, y_test, int(n_classes))
    return training, test


def import_IB_data_from_mat(name_ID, nb_dir = ''):
    """ Writes a .npy file to disk containing the harmonics dataset used by Tishby
    Args:
        name_ID: Identifier which is going to be part of the output filename

    Returns:
        None
    """
    print('Loading Data...')
    d = sio.loadmat(nb_dir + 'datasets/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets_original = C()
    data_sets_original.data = F
    data_sets_original.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

    data_sets = utils.data_shuffle(data_sets_original, 80, shuffle_data=True)
    X_train, y_train, X_test, y_test = data_sets.train.data, data_sets.train.labels[:,
                                                             0], data_sets.test.data, data_sets.test.labels[:, 0]
    np.savez_compressed(nb_dir + 'datasets/IB_data_' + str(name_ID), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
