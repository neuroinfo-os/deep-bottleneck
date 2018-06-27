import numpy as np

from iclr_wrap_up.mi_estimator.base import MutualInformationEstimator


def load(training_data, test_data, architecture, full_mi):
    estimator = BinningMutualInformationEstimator(training_data, test_data, architecture, full_mi)
    return estimator


class BinningMutualInformationEstimator(MutualInformationEstimator):

    def __init__(self, training_data, test_data, architecture, full_mi):
        super().__init__(training_data, test_data, architecture, full_mi)
        self.binsize = 0.07

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