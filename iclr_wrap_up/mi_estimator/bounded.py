from iclr_wrap_up.mi_estimator.base import MutualInformationEstimator


class BoundedMutualInformationEstimator(MutualInformationEstimator):

    def __init__(self, training_data, test_data, architecture, full_mi):
        super().__init__(training_data, test_data, architecture, full_mi)
        self.noise_variance = 1e-3  # Added Gaussian noise variance.

    def _estimate_entropy(self, data):
        return self._K_estimate_entropy(data)[0]
