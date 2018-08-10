import numpy as np

from deep_bottleneck.mi_estimator.base import MutualInformationEstimator


def load(discretization_range, training_data, test_data, architecture, calculate_mi_for):
    estimator = BinningMutualInformationEstimator(discretization_range, training_data, test_data, architecture, calculate_mi_for)
    return estimator


class BinningMutualInformationEstimator(MutualInformationEstimator):

    def __init__(self, discretization_range, training_data, test_data, architecture, calculate_mi_for):
        super().__init__(discretization_range, training_data, test_data, architecture, calculate_mi_for)
        self.binsize = discretization_range

    def _estimate_entropy(self, data):
        digitized = np.floor(data / self.binsize).astype('int')
        p_ts, _ = self._get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    def _get_unique_probs(self, x):
        # TODO what happens here?
        uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True,
                                                     return_counts=True)
        return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

    def _estimate_conditional_entropy(self, data: np.array) -> float:
        return 0
