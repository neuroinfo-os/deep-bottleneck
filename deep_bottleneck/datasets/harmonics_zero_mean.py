from deep_bottleneck.datasets.base_dataset import Dataset
from deep_bottleneck.datasets.harmonics import load as load_original_harmonics


def load(nb_dir='') -> Dataset:
    """Load the Information Bottleneck harmonics dataset with mean 0, i.e.
    the original harmonics dataset but 0.5 subtracted

    Returns:
        The zero centered harmonics dataset.
    """

    data = load_original_harmonics()

    X_train = data.train.examples - 0.5
    y_train = data.train.labels
    X_test = data.test.examples - 0.5
    y_test = data.test.labels
    n_classes = data.n_classes

    shifted_dataset = Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)

    return shifted_dataset
