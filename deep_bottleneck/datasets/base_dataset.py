from tensorflow.python.keras import utils as keras_utils


class Dataset:
    """Representation of a dataset."""

    @classmethod
    def from_labelled_subset(cls, X_train, y_train, X_test, y_test, n_classes):
        training_set = LabelledDataset.from_labels(X_train, y_train, n_classes)
        test_set = LabelledDataset.from_labels(X_test, y_test, n_classes)

        return cls(training_set, test_set, n_classes)

    def __init__(self, train, test, n_classes):
        self.train = train
        self.test = test
        self.n_classes = n_classes


class LabelledDataset:
    """Representation of a labelled subset of a dataset.

    This could be a training, test or validation set.
    """

    @classmethod
    def from_labels(cls, examples, labels, n_classes):
        one_hot_labels = keras_utils.to_categorical(labels, n_classes).astype('float32')
        return cls(examples, labels, one_hot_labels)

    def __init__(self, examples, labels, one_hot_labels):
        self.examples = examples
        self.labels = labels
        self.one_hot_labels = one_hot_labels
