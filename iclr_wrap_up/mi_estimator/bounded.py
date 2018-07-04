from iclr_wrap_up.mi_estimator.base import MutualInformationEstimator
from iclr_wrap_up.mi_estimator import kde

class BoundedMutualInformationEstimator(MutualInformationEstimator):

    def __init__(self, training_data, test_data, architecture, calculate_mi_for):
        super().__init__(training_data, test_data, architecture, calculate_mi_for)
        self.noise_variance = 1e-3  # Added Gaussian noise variance.

    def _estimate_entropy(self, data):
        return self._K_estimate_entropy([data])[0]

    def _estimate_conditional_entropy(self, data):
        return kde.kde_condentropy(data, self.noise_variance)