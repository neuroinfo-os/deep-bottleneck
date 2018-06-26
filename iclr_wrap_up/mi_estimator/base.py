import pandas as pd
import numpy as np


class MutualInformationEstimator:
    nats2bits = 1.0 / np.log(2)
    """Nats to bits conversion factor."""

    def __init__(self, training_data, test_data, epochs, architecture_name, full_mi, activation_fn):
        self.training_data = training_data
        self.test_data = test_data
        self.epochs = epochs
        self.architecture = architecture_name
        self.full_mi = full_mi
        self.activation_fn = activation_fn

    def compute_mi(self, activations_summary) -> pd.DataFrame:
        raise NotImplementedError
