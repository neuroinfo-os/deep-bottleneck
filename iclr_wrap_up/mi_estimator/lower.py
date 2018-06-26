from collections import OrderedDict

import pandas as pd
import numpy as np

from iclr_wrap_up.mi_estimator import kde
from iclr_wrap_up.mi_estimator.base import MutualInformationEstimator
from iclr_wrap_up import utils

from tensorflow.python.keras import backend as K
from operator import itemgetter

from typing import *


def load(training_data, test_data, architecture, full_mi):
    estimator = LowerBoundMutualInformationEstimator(training_data, test_data, architecture, full_mi)
    return estimator


class LowerBoundMutualInformationEstimator(MutualInformationEstimator):

    def __init__(self, training_data, test_data, architecture, full_mi):
        self.training_data = training_data
        self.test_data = test_data
        self.architecture = architecture
        self.full_mi = full_mi
        self.noise_variance = 1e-3  # Added Gaussian noise variance.
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder.
        self._K_estimate_entropy = K.function([Klayer_activity, ],
                                        [kde.entropy_estimator_bd(Klayer_activity,
                                                                  self.noise_variance)])

    def compute_mi(self, epoch_summaries: OrderedDict):
        """Return lower bound on entropy of layer activity.

        Args:
            epoch_summaries: An Orde
        """
        print(f'*** Start running {self.__class__.__name__}. ***')

        labels, one_hot_labels = self._construct_dataset()
        # Proportion of instances that has a certain label.
        labelprobs = np.mean(one_hot_labels, axis=0)
        measures = self._init_dataframe(epoch_numbers=epoch_summaries.keys())
        n_layers = len(self.architecture) + 1

        for epoch, summary in epoch_summaries.items():

            for layer_index in range(n_layers):
                mi_with_input, mi_with_output = self._compute_mi_per_epoch_and_layer(labelprobs, layer_index,
                                                                                     labels, summary)

                measures.loc[(epoch, layer_index), 'MI_XM'] = mi_with_input
                measures.loc[(epoch, layer_index), 'MI_YM'] = mi_with_output

        return measures

    def _compute_mi_per_epoch_and_layer(self, labelprobs, layer_index, labels, summary) -> Tuple[float, float]:
        activity = summary['data']['activity_tst'][layer_index]
        h_lower = self._estimate_entropy([activity])
        # Layer activity given input. This is simply the entropy of the Gaussian noise.
        hM_given_X = kde.kde_condentropy(activity, self.noise_variance)
        hM_given_Y_lower = self._compute_hM_given_Y_lower(activity, labelprobs, labels)
        mi_with_input = self.nats2bits * (h_lower - hM_given_X)
        mi_with_output = self.nats2bits * (h_lower - hM_given_Y_lower)
        return mi_with_input, mi_with_output

    def _compute_hM_given_Y_lower(self, activity, labelprobs, labels) -> float:
        hM_given_Y_lower = 0
        for target_class in range(int(self.training_data.n_classes)):
            hcond_lower = self._estimate_entropy([activity[labels == target_class]])
            hM_given_Y_lower += labelprobs[target_class] * hcond_lower
        return hM_given_Y_lower

    def _init_dataframe(self, epoch_numbers):
        info_measures = ['MI_XM', 'MI_YM']
        n_layers = len(self.architecture) + 1  # + 1 for output layer
        index_base_keys = [epoch_numbers, list(range(n_layers))]
        index = pd.MultiIndex.from_product(index_base_keys, names=['epoch', 'layer'])
        measures = pd.DataFrame(index=index, columns=info_measures)
        return measures

    def _construct_dataset(self):
        # Y is a one-hot vector, y is a label vector.
        if self.full_mi:
            full = utils.construct_full_dataset(self.training_data, self.test_data)
            labels = full.y
            one_hot_labels = full.Y
        else:
            labels = self.test_data.y
            one_hot_labels = self.test_data.Y

        return labels, one_hot_labels

    def _estimate_entropy(self, data):
        return self._K_estimate_entropy(data)[0]

