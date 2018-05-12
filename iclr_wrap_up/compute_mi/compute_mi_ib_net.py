import os
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
from tensorflow.contrib.keras import backend as K

from iclr_wrap_up import kde
from iclr_wrap_up import simplebinmi
from iclr_wrap_up import utils

import seaborn as sns

sns.set_style('darkgrid')


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

    def compute_mi(self):
        # Which measure to plot
        # infoplane_measure = 'bin'

        DO_LOWER = (self.infoplane_measure == 'lower')  # Whether to compute lower bounds also
        DO_BINNED = (self.infoplane_measure == 'bin')  # Whether to compute MI estimates based on binning


        # Directories from which to load saved layer activity
        DIR_TEMPLATE = '%%s_%s' % self.architecture_name

        # Functions to return upper and lower bounds on entropy of layer activity
        noise_variance = 1e-3  # Added Gaussian noise variance
        binsize = 0.07  # size of bins for binning method
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder
        entropy_func_upper = K.function([Klayer_activity, ],
                                        [kde.entropy_estimator_kl(Klayer_activity, noise_variance), ])
        entropy_func_lower = K.function([Klayer_activity, ],
                                        [kde.entropy_estimator_bd(Klayer_activity, noise_variance), ])

        # nats to bits conversion factor
        nats2bits = 1.0 / np.log(2)

        # Save indexes of tests data for each of the output classes
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

        # ------------------------------------

        PLOT_LAYERS = None  # Which layers to plot.  If None, all saved layers are plotted

        # Data structure used to store results
        measures = OrderedDict()
        measures[self.activation_fn] = {}

        # ----------------------------------------

        for activation in measures.keys():
            cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
            if not os.path.exists(cur_dir):
                print("Directory %s not found" % cur_dir)
                continue

        # Load files saved during each epoch, and compute MI measures of the activity in that epoch
        print('*** Doing %s ***' % cur_dir)
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue

            fname = cur_dir + "/" + epochfile
            with open(fname, 'rb') as f:
                d = pickle.load(f)

            epoch = d['epoch']
            if epoch in measures[activation]:  # Skip this epoch if its already been processed
                continue  # this is a trick to allow us to rerun this cell multiple times)

            if epoch > self.epochs:
                continue

            print("Doing", fname)

            num_layers = len(d['data']['activity_tst'])

            if PLOT_LAYERS is None:
                PLOT_LAYERS = []
                for lndx in range(num_layers):
                    # if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                    PLOT_LAYERS.append(lndx)

            cepochdata = defaultdict(list)
            for lndx in range(num_layers):
                activity = d['data']['activity_tst'][lndx]

                # Compute marginal entropies
                h_upper = entropy_func_upper([activity, ])[0]
                if DO_LOWER:
                    h_lower = entropy_func_lower([activity, ])[0]

                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde.kde_condentropy(activity, noise_variance)

                # Compute conditional entropies of layer activity given output
                hM_given_Y_upper = 0.
                for i in range(self.training_data.nb_classes):
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
                cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
                cepochdata['H_M_upper'].append(nats2bits * h_upper)

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                    cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])

                if DO_LOWER:  # Compute lower bounds

                    hM_given_Y_lower = 0.
                    for i in range(self.training_data.nb_classes):
                        hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])[0]
                        hM_given_Y_lower += labelprobs[i] * hcond_lower

                    cepochdata['MI_XM_lower'].append(nats2bits * (h_lower - hM_given_X))
                    cepochdata['MI_YM_lower'].append(nats2bits * (h_lower - hM_given_Y_lower))
                    cepochdata['H_M_lower'].append(nats2bits * h_lower)
                    pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                if DO_BINNED:  # Compute binned estimates
                    binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, binsize)
                    cepochdata['MI_XM_bin'].append(nats2bits * binxm)
                    cepochdata['MI_YM_bin'].append(nats2bits * binym)
                    pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                        cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])

                print('- Layer %d %s' % (lndx, pstr))

            measures[activation][epoch] = cepochdata

        return measures, PLOT_LAYERS


