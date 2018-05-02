import os
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
import keras.backend as K

from iclr_wrap_up import kde
from iclr_wrap_up import simplebinmi
from iclr_wrap_up import utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')


def load(training_data, test_data, epochs, architecture_name, full_mi):
    estimator = MutualInformationEstimator(training_data, test_data, epochs, architecture_name, full_mi)
    return estimator



class MutualInformationEstimator:

    def __init__(self, training_data, test_data, epochs, architecture_name, full_mi):
        self.training_data = training_data
        self.test_data = test_data
        self.epochs = epochs
        self.architecture_name = architecture_name
        self.full_mi = full_mi




    def compute_mi(self):
        # Which measure to plot
        infoplane_measure = 'upper'
        # infoplane_measure = 'bin'

        DO_LOWER       = (infoplane_measure == 'lower')   # Whether to compute lower bounds also
        DO_BINNED      = (infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning

        NUM_LABELS = 2
        # MAX_EPOCHS = 1000
        COLORBAR_MAX_EPOCHS = 10000

        # Directories from which to load saved layer activity
        # ARCH = '1024-20-20-20'
        architecture_name = '10-7-5-4-3'
        #ARCH = '20-20-20-20-20-20'
        #ARCH = '32-28-24-20-16-12'
        #ARCH = '32-28-24-20-16-12-8-8'
        DIR_TEMPLATE = '%%s_%s'%architecture_name

        # Functions to return upper and lower bounds on entropy of layer activity
        noise_variance = 1e-3                    # Added Gaussian noise variance
        binsize = 0.07                           # size of bins for binning method
        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder
        entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
        entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

        # nats to bits conversion factor
        nats2bits = 1.0/np.log(2)

        # Save indexes of tests data for each of the output classes
        saved_labelixs = {}

        y = self.test_data.y
        Y = self.test_data.Y
        if self.full_mi:
            full = utils.construct_full_dataset(self.training_data, self.test_data)
            y = full.y
            Y = full.Y

        for i in range(NUM_LABELS):
            saved_labelixs[i] = y == i

        labelprobs = np.mean(Y, axis=0)


        # ------------------------------------

        PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted

        # Data structure used to store results
        measures = OrderedDict()
        measures['tanh'] = {}
        measures['relu'] = {}
        # measures['softsign'] = {}
        # measures['softplus'] = {}

        #----------------------------------------

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
                for i in range(NUM_LABELS):
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                if DO_LOWER:
                    hM_given_Y_lower = 0.
                    for i in range(NUM_LABELS):
                        hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])[0]
                        hM_given_Y_lower += labelprobs[i] * hcond_lower

                cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
                cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
                cepochdata['H_M_upper'].append(nats2bits * h_upper)

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                if DO_LOWER:  # Compute lower bounds
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

        #--------------------------

        max_epoch = max((max(vals.keys()) if len(vals) else 0) for vals in measures.values())
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
        sm._A = []

        fig = plt.figure(figsize=(10, 5))
        for actndx, (activation, vals) in enumerate(measures.items()):
            epochs = sorted(vals.keys())
            if not len(epochs):
                continue
            plt.subplot(1, 2, actndx + 1)
            for epoch in epochs:
                c = sm.to_rgba(epoch)
                xmvals = np.array(vals[epoch]['MI_XM_' + infoplane_measure])[PLOT_LAYERS]
                ymvals = np.array(vals[epoch]['MI_YM_' + infoplane_measure])[PLOT_LAYERS]

                plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
                plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

            plt.ylim([0, 1])
            plt.xlim([0, 12])
            #     plt.ylim([0, 3.5])
            #     plt.xlim([0, 14])
            plt.xlabel('I(X;M)')
            plt.ylabel('I(Y;M)')
            plt.title(activation)

        cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
        plt.colorbar(sm, label='Epoch', cax=cbaxes)
        plt.tight_layout()

        # if DO_SAVE:
        # plt.savefig('plots/' + DIR_TEMPLATE % ('infoplane_'+ARCH),bbox_inches='tight')

        #------------------------------------------------

        plt.figure(figsize=(12, 5))

        gs = gridspec.GridSpec(len(measures), len(PLOT_LAYERS))
        for activation in measures.keys():
            cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
            if not os.path.exists(cur_dir):
                continue

            epochs = []
            means = []
            stds = []
            wnorms = []
            for epochfile in sorted(os.listdir(cur_dir)):
                if not epochfile.startswith('epoch'):
                    continue

                with open(cur_dir + "/" + epochfile, 'rb') as f:
                    d = pickle.load(f)

                epoch = d['epoch']
                epochs.append(epoch)
                wnorms.append(d['data']['weights_norm'])
                means.append(d['data']['gradmean'])
                stds.append(d['data']['gradstd'])

            wnorms, means, stds = map(np.array, [wnorms, means, stds])
            for lndx, layerid in enumerate(PLOT_LAYERS):
                plt.subplot(gs[actndx, lndx])
                plt.plot(epochs, means[:, layerid], 'b', label="Mean")
                plt.plot(epochs, stds[:, layerid], 'orange', label="Std")
                plt.plot(epochs, means[:, layerid] / stds[:, layerid], 'red', label="SNR")
                plt.plot(epochs, wnorms[:, layerid], 'g', label="||W||")

                plt.title('Layer %d' % layerid)
                plt.xlabel('Epoch')
                plt.gca().set_xscale("log", nonposx='clip')
                plt.gca().set_yscale("log", nonposy='clip')

        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
        plt.tight_layout()

        plt.savefig('plots/' + DIR_TEMPLATE % ('snr_'+architecture_name), bbox_inches='tight')

