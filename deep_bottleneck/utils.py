import os
import numpy as np
from collections import namedtuple
from dotenv import load_dotenv
from pathlib import Path

def construct_full_dataset(training, test):
    """Concatenates training and test data splits to obtain the full dataset.

    The input arguments use the following naming convention:
        - X is the training data
        - y is training class, with numbers from 0 to 1
        - Y is training class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
  
    Args:
        training: Namedtuple with fields X, y and Y:
        test: Namedtuple with fields X, y and Y:

    Returns:
        A new Namedtuple with fields X, y and Y containing the concatenation of training and test data
    """
    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'n_classes'])
    X = np.concatenate((training.X, test.X))
    y = np.concatenate((training.y, test.y))
    Y = np.concatenate((training.Y, test.Y))
    return Dataset(X, Y, y, training.n_classes)


def shuffle_in_unison_inplace(a, b):
    """Shuffles both array a and b randomly in unison
    Args:
        a: An Array, for example containing data samples
        b: An Array, fpor example containing labels

    Returns:
        Both arrays shuffled in the same way
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    # TODO Function data_shuffle need refctoring and proper docstring
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
    """Check whether a layer has attribute 'kernel', which is true for dense-like layers
    Args:
        layer: Keras layer to check for attribute 'kernel'

    Returns:
        True if layer has attribute 'kernel', False otherwise
    """
    return hasattr(layer, 'kernel')


def _get_current_min_max(activations):
    """Get both minimum and maximum of an array
    Args:
        activations: numpy ndarray

    Returns:
        Minimum and maximum value of activations
    """
    return np.min(activations), np.max(activations)

def get_min_max(activations_summary, layer_number, neuron_number=None):
    """Get minimum and maximum of activations of a specific layer or a specific neuron over all epochs
    Args:
        activations_summary: numpy ndarray
        layer_number: Index of the layer
        neuron_number: Index of the neuron. If None, activations of the whole layer serve as a basis

    Returns:
        Minimum and maximum value of activations over all epochs
    """
    epochs_in_activation_summary = [int(epoch) for epoch in activations_summary]
    epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))

    total_max = float("-inf")
    total_min = float("inf")
    for epochs in epochs_in_activation_summary:
        activations = activations_summary[f'{epochs}/activations']
        layer_activations = np.asarray(activations[str(layer_number)])
        layer_activations = layer_activations.transpose()

        if neuron_number is not None:
            current_min, current_max = _get_current_min_max(layer_activations[neuron_number])
        else:
            current_min, current_max = _get_current_min_max(layer_activations)

        total_max = np.max([current_max, total_max])
        total_min = np.min([current_min, total_min])

    return total_min, total_max

ENV_PATH = Path(__file__).parent.parent  / "infrastructure" / "sacred_setup" / ".env"

def get_mongo_config():
    load_dotenv(dotenv_path=ENV_PATH)
    uri = (
            f'mongodb://{os.environ["MONGO_INITDB_ROOT_USERNAME"]}:'
            f'{os.environ["MONGO_INITDB_ROOT_PASSWORD"]}@{os.environ["MONGO_HOST"]}/?authMechanism=SCRAM-SHA-1'
        )

    return uri, os.environ["MONGO_DATABASE"]