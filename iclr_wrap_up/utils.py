import numpy as np
from collections import namedtuple


def construct_full_dataset(training, test):
    """Concatenates training and test data splits to obtain the full dataset.
    Args:
        training: Namedtuple with fields X, y and Y:
            - X is the training data
            - y is training class, with numbers from 0 to 1
            - Y is training class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class
        test: Namedtuple with fields X, y and Y:
            - X is the test data
            - y is test class, with numbers from 0 to 1
            - Y is test class, but coded as a 2-dim vector with one entry set to 1 at the column index corresponding to the class

    Returns:
        A new Namedtuple with fields X, y and Y containing the concatenation of training and test data
    """
    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'n_classes'])
    X = np.concatenate((training.X, test.X))
    y = np.concatenate((training.y, test.y))
    Y = np.concatenate((training.Y, test.Y))
    return Dataset(X, Y, y, training.n_classes)


def shuffle_in_unison_inplace(a, b):
    """ Shuffles both array a and b randomly in unison
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
    """ Divided the data to train and test and shuffle it """
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


def get_min_max(activations_summary, layer_number, neuron_number=None):
    epochs_in_activation_summary = [int(epoch) for epoch in activations_summary]
    epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))

    total_max = 0
    total_min = 0
    for epochs in epochs_in_activation_summary:
        activations = activations_summary[f'{epochs}/activations']

        if neuron_number is not None:
            layer_activations = np.asarray(activations[str(layer_number)])
            layer_activations = layer_activations.transpose()

            current_max = np.max(layer_activations[neuron_number])
            current_min = np.min(layer_activations[neuron_number])
        else:
            layer_activations = np.asarray(activations[str(layer_number)])

            current_max = np.max(layer_activations)
            current_min = np.min(layer_activations)

        if current_max > total_max:
            total_max = current_max
        if current_min < total_min:
            total_min = current_min

    return total_min, total_max