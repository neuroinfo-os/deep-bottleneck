from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from deep_bottleneck.datasets.base_dataset import Dataset


def load():
    """Load a randomly generated classification dataset."""
    n_classes = 2

    X, y = make_classification(
        n_samples=5_000,
        n_features=12,
        n_informative=6,
        n_redundant=4,
        n_classes=n_classes,
        flip_y=0.1,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)

