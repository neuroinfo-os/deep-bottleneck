import os
import pickle

import pandas as pd
import numpy as np
from tensorflow.python.keras import backend as K

from iclr_wrap_up import kde
from iclr_wrap_up import simplebinmi
from iclr_wrap_up import utils

def load(training_data, test_data, epochs, architecture_name, full_mi, activation_fn, infoplane_measure):
    estimator = MutualInformationEstimator(training_data, test_data, epochs,
                                           architecture_name, full_mi, activation_fn, infoplane_measure)
    return estimator


class MutualInformationEstimator:

    def __init__(self, training_data, test_data, epochs, architecture_name, full_mi, activation_fn, infoplane_measure):
        self.training_data = training_data
        self.test_data = test_data
        self.epochs = epochs
        self.architecture_name = architecture_name
        self.full_mi = full_mi
        self.activation_fn = activation_fn
        self.infoplane_measure = infoplane_measure

    def compute_mi(self, activations_summary):

        binsize = 0.07  # Size of bins for binning method.
        numbins = 100   # Number of bins for other binning method.

        # Functions to return upper and lower bounds on entropy of layer activity.
        noise_variance = 1e-3  # Added Gaussian noise variance.

        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder.
        entropy_func_upper = K.function([Klayer_activity, ],
                                        [kde.entropy_estimator_kl(Klayer_activity, noise_variance), ])
        entropy_func_lower = K.function([Klayer_activity, ],
                                        [kde.entropy_estimator_bd(Klayer_activity, noise_variance), ])

        # Nats to bits conversion factor.
        nats2bits = 1.0 / np.log(2)

        # Save indexes of tests data for each of the output classes.
        saved_labelixs = {}

        y = self.test_data.y
        Y = self.test_data.Y
        if self.full_mi:
            full = utils.construct_full_dataset(self.training_data, self.test_data)
            y = full.y
            Y = full.Y

        for i in range(self.training_data.nb_classes):
            saved_labelixs[i] = y == i

        labelprobs = np.mean(Y, axis=0)

        info_measures = ['MI_XM_upper', 'MI_YM_upper', 'MI_XM_lower', 'MI_YM_lower', 'MI_XM_bin',  'MI_XM_bin2',
                         'MI_YM_bin', 'MI_YM_bin2', 'H_M_upper', 'H_M_lower']


        # Build up index and Data structure to store results.
        epoch_nrs = []
        for s in activations_summary.keys():
            epoch_nrs.append(int((s[5:].lstrip('0'))))

        epoch_nrs.sort()

        num_layers = len(self.architecture_name.split('-'))

        index_base_keys = [epoch_nrs, list(range(num_layers))]
        index = pd.MultiIndex.from_product(index_base_keys, names=['epoch', 'layer'])

        measures = pd.DataFrame(index=index, columns=info_measures)

        # Load files saved during each epoch, and compute MI measures of the activity in that epoch
        print(f'*** Start Iterations over epochs ***')
        for epoch_number, epoch_values in activations_summary.items():

            print('Doing epoch nr.: ', epoch_number)
            epoch = epoch_values['epoch']

            if epoch > self.epochs:
                continue

            num_layers = len(epoch_values['data']['activity_tst'])

            for layer_index in range(num_layers):
                activity = epoch_values['data']['activity_tst'][layer_index]

                if self.infoplane_measure == "upper":
                    # Compute marginal entropies
                    h_upper = entropy_func_upper([activity, ])[0]

                    # Layer activity given input. This is simply the entropy of the Gaussian noise
                    hM_given_X = kde.kde_condentropy(activity, noise_variance)

                    # Compute conditional entropies of layer activity given output
                    hM_given_Y_upper = 0.
                    for i in range(self.training_data.nb_classes):
                        hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])[0]
                        hM_given_Y_upper += labelprobs[i] * hcond_upper

                    measures.loc[(epoch, layer_index), 'MI_XM_upper'] = nats2bits * (h_upper - hM_given_X)
                    measures.loc[(epoch, layer_index), 'MI_YM_upper'] = nats2bits * (h_upper - hM_given_Y_upper)
                    measures.loc[(epoch, layer_index), 'H_M_upper'] = nats2bits * h_upper

                    pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        measures.loc[(epoch, layer_index), 'MI_XM_upper'], measures.loc[(epoch, layer_index), 'MI_YM_upper'])

                if self.infoplane_measure == "lower":

                    h_lower = entropy_func_lower([activity, ])[0]

                    # Layer activity given input. This is simply the entropy of the Gaussian noise.
                    hM_given_X = kde.kde_condentropy(activity, noise_variance)

                    hM_given_Y_lower = 0.

                    for i in range(self.training_data.nb_classes):
                        hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])[0]
                        hM_given_Y_lower += labelprobs[i] * hcond_lower

                    measures.loc[(epoch, layer_index), 'MI_XM_lower'] = nats2bits * (h_lower - hM_given_X)
                    measures.loc[(epoch, layer_index), 'MI_YM_lower'] = nats2bits * (h_lower - hM_given_Y_lower)
                    measures.loc[(epoch, layer_index), 'H_M_lower'] = nats2bits * h_lower

                    pstr = ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        measures.loc[(epoch, layer_index), 'MI_XM_lower'], measures.loc[(epoch, layer_index), 'MI_YM_lower'])

                if self.infoplane_measure == "bin":
                    binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, binsize)
                    measures.loc[(epoch, layer_index), 'MI_XM_bin'] = nats2bits * binxm
                    measures.loc[(epoch, layer_index), 'MI_YM_bin'] = nats2bits * binym

                    pstr = ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        measures.loc[(epoch, layer_index), 'MI_XM_bin'], measures.loc[(epoch, layer_index), 'MI_YM_bin'])

                if self.infoplane_measure == "bin2":
                    binxm, binym = simplebinmi.bin_calc_information_evenbins(saved_labelixs, activity, numbins)
                    measures.loc[(epoch, layer_index), 'MI_XM_bin'] = nats2bits * binxm
                    measures.loc[(epoch, layer_index), 'MI_YM_bin'] = nats2bits * binym

                    pstr = ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        measures.loc[(epoch, layer_index), 'MI_XM_bin'], measures.loc[(epoch, layer_index), 'MI_YM_bin'])


                print(f'- Layer {layer_index} {pstr}')

        return measures
