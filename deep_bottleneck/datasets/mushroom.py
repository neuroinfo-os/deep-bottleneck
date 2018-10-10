import pandas as pd
from sklearn.model_selection import train_test_split
from deep_bottleneck.datasets.base_dataset import Dataset


def load():
    """Load the mushroom dataset.
    
    Mushrooms are to be classified as either edible or poisonous.

    Returns:
        The mushroom dataset.
    """
    mushrooms = pd.read_csv('datasets/mushroom.csv')

    n_classes = 2
    y = mushrooms['class=e']
    X = mushrooms.drop(['class=e', 'class=p'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    dataset = Dataset.from_labelled_subset(X_train, y_train, X_test, y_test, n_classes)

    return dataset
