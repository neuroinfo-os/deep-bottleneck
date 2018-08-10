from deep_bottleneck.mi_estimator import kde
from deep_bottleneck.mi_estimator.bounded import BoundedMutualInformationEstimator

from tensorflow.python.keras import backend as K


def load(discretization_range, training_data, test_data, architecture, calculate_mi_for):
    estimator = LowerBoundMutualInformationEstimator(discretization_range, training_data, test_data, architecture, calculate_mi_for)
    return estimator


class LowerBoundMutualInformationEstimator(BoundedMutualInformationEstimator):

    def __init__(self, discretization_range, training_data, test_data, architecture, calculate_mi_for):
        super().__init__(discretization_range, training_data, test_data, architecture, calculate_mi_for)
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder.
        self._K_estimate_entropy = K.function([Klayer_activity],
                                              [kde.entropy_estimator_bd(Klayer_activity,
                                                                        self.noise_variance)])
