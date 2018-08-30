from pathlib2 import Path
import scipy.io as sio
import numpy as np

from deep_bottleneck import utils
from deep_bottleneck.datasets.base_dataset import Dataset


def load(nb_dir='') -> Dataset:
    """Load the Information Bottleneck harmonics dataset

    Returns:
        The harmonics dataset.
    """
    ID = '2017_12_21_16_51_3_275766'
    n_classes = 2
    data_file = Path(nb_dir + 'datasets/IB_data_' + str(ID) + '.npz')
    if not data_file.is_file():
        import_IB_data_from_mat(ID, nb_dir)

    data = np.load(nb_dir + 'datasets/IB_data_' + str(ID) + '.npz')

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    dataset = Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)

    return dataset


def import_IB_data_from_mat(name_ID, nb_dir=''):
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
    np.savez_compressed(nb_dir + 'datasets/IB_data_' + str(name_ID), X_train=X_train, y_train=y_train, X_test=X_test,
                        y_test=y_test)
