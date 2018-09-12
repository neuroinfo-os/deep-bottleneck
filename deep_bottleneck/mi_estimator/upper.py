from deep_bottleneck.mi_estimator import kde
from deep_bottleneck.mi_estimator.bounded import BoundedMutualInformationEstimator

from tensorflow.python.keras import backend as K


def load(discretization_range, architecture, n_classes):
    estimator = UpperBoundMutualInformationEstimator(discretization_range, architecture, n_classes)
    return estimator


class UpperBoundMutualInformationEstimator(BoundedMutualInformationEstimator):

    def __init__(self, discretization_range, architecture, n_classes):
        super().__init__(discretization_range, architecture, n_classes)
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder.
        self._K_estimate_entropy = K.function([Klayer_activity],
                                              [kde.entropy_estimator_kl(Klayer_activity,
                                                                        self.noise_variance)])
