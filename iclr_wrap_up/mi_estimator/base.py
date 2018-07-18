from typing import *

import pandas as pd
import numpy as np

from iclr_wrap_up import utils

from iclr_wrap_up.mi_estimator import kde

class MutualInformationEstimator:
    nats2bits = 1.0 / np.log(2)
    """Nats to bits conversion factor."""

    def __init__(self, discretization_range, training_data, test_data, architecture, calculate_mi_for):
        self.training_data = training_data
        self.test_data = test_data
        self.architecture = architecture
        self.calculate_mi_for = calculate_mi_for

    def compute_mi(self, file_all_activations) -> pd.DataFrame:
        print(f'*** Start running {self.__class__.__name__}. ***')

        print(file_all_activations["2"])
        print(f'len of file activations: {len(file_all_activations)}')
        for i in file_all_activations: print(file_all_activations[str(i)])

        labels, one_hot_labels = self._construct_dataset()
        # Proportion of instances that have a certain label.
        label_weights = np.mean(one_hot_labels, axis=0)
        label_masks = {}
        for target_class in range(self.training_data.n_classes):
            label_masks[target_class] = labels == target_class
        n_layers = len(self.architecture) + 1  # + 1 for output layer
        epoch_numbers = [int(value) for value in file_all_activations]
        epoch_numbers = sorted(epoch_numbers)
        measures = self._init_dataframe(epoch_numbers=epoch_numbers, n_layers=n_layers)

        #for epoch, summary in epoch_summaries.items():
        for epoch in file_all_activations:
            print(f'Estimating mutual information for epoch {epoch}.')
            summary = file_all_activations[epoch]
            epoch = int(epoch)
            for layer_index in range(n_layers):
                layer_activations = summary['activations'][str(layer_index)]
                mi_with_input, mi_with_label = self._compute_mi_per_epoch_and_layer(layer_activations, label_weights,
                                                                                    label_masks)

                measures.loc[(epoch, layer_index), 'MI_XM'] = mi_with_input
                measures.loc[(epoch, layer_index), 'MI_YM'] = mi_with_label
        return measures

    def _construct_dataset(self):
        # Y is a one-hot vector, y is a label vector.
        if self.calculate_mi_for == "full_dataset":
            full = utils.construct_full_dataset(self.training_data, self.test_data)
            labels = full.y
            one_hot_labels = full.Y
        elif self.calculate_mi_for == "test":
            labels = self.test_data.y
            one_hot_labels = self.test_data.Y
        elif self.calculate_mi_for == "training":
            labels = self.training_data.y
            one_hot_labels = self.training_data.Y

        return labels, one_hot_labels

    def _init_dataframe(self, epoch_numbers, n_layers):
        info_measures = ['MI_XM', 'MI_YM']
        index_base_keys = [epoch_numbers, list(range(n_layers))]
        index = pd.MultiIndex.from_product(index_base_keys, names=['epoch', 'layer'])
        measures = pd.DataFrame(index=index, columns=info_measures)
        return measures

    def _compute_mi_per_epoch_and_layer(self, activations, label_weights, label_masks) -> Tuple[float, float]:
        activations = np.asarray(activations)
        H_of_M = self._estimate_entropy(activations)
        H_of_M_given_X = self._estimate_conditional_entropy(activations)
        H_of_M_given_Y = self._compute_H_of_M_given_Y(activations, label_weights, label_masks)
        mi_with_input = self.nats2bits * (H_of_M - H_of_M_given_X)
        mi_with_label = self.nats2bits * (H_of_M - H_of_M_given_Y)

        return mi_with_input, mi_with_label

    def _compute_H_of_M_given_Y(self, activations, label_weights, label_masks):
        H_of_M_given_Y = 0
        for label, mask in label_masks.items():
            H_of_M_for_specific_y = self._estimate_entropy(activations[mask])
            H_of_M_given_Y += label_weights[label] * H_of_M_for_specific_y
        return H_of_M_given_Y

    def _estimate_entropy(self, data: np.array) -> float:
        """

        Args:
            data: The data to estimate entropy for.

        Returns:
            The estimated entropy.
        """
        raise NotImplementedError

    def _estimate_conditional_entropy(self, data: np.array) -> float:
        raise NotImplementedError