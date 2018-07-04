from iclr_wrap_up.mi_estimator import kde
from iclr_wrap_up.mi_estimator.bounded import BoundedMutualInformationEstimator

from tensorflow.python.keras import backend as K


def load(discretization_range, training_data, test_data, architecture, full_mi):
    estimator = UpperBoundMutualInformationEstimator(discretization_range, training_data, test_data, architecture, full_mi)
    return estimator


class UpperBoundMutualInformationEstimator(BoundedMutualInformationEstimator):

    def __init__(self, discretization_range, training_data, test_data, architecture, full_mi):
        super().__init__(discretization_range, training_data, test_data, architecture, full_mi)
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder.
        self._K_estimate_entropy = K.function([Klayer_activity],
                                              [kde.entropy_estimator_kl(Klayer_activity,
                                                                        self.noise_variance)])
