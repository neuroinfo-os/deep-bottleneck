import matplotlib.pyplot as plt
import numpy as np

from iclr_wrap_up.plotter.base import BasePlotter


def load(run, dataset, architecture):
    return SignalToNoiseRationPlotter(run, dataset, architecture)


# TODO think about whether plotting snr ratio averaged over multiple runs does make sense

class SignalToNoiseRationPlotter(BasePlotter):
    plotname = 'snr'

    def __init__(self, run, dataset, architecture):
        self.dataset = dataset
        self.run = run
        self.architecture = architecture

    def plot(self, measures_summary):

        activations_summary = measures_summary['activations_summary']

        epochs = []
        means = []
        stds = []
        wnorms = []

        for epoch_number, epoch_values in activations_summary.items():
            epoch = epoch_values['epoch']
            epochs.append(epoch)
            wnorms.append(epoch_values['data']['weights_norm'])
            means.append(epoch_values['data']['gradmean'])
            stds.append(epoch_values['data']['gradstd'])

        wnorms, means, stds = map(np.array, [wnorms, means, stds])
        plot_layers = range(len(self.architecture) + 1)  # +1 for the last output layer.

        fig = plt.figure(figsize=(12, 5))

        for lndx, layerid in enumerate(plot_layers):
            ax = fig.add_subplot(1, len(plot_layers), lndx + 1)

            ax.plot(epochs, means[:, layerid], 'b', label='Mean')
            ax.plot(epochs, stds[:, layerid], 'orange', label='Std')
            ax.plot(epochs, means[:, layerid] / stds[:, layerid], 'red', label='SNR')
            ax.plot(epochs, wnorms[:, layerid], 'g', label='||W||')

            ax.set_title(f'Layer {layerid}')
            ax.set_xlabel('Epoch')
            ax.set_xscale('log', nonposx='clip')
            ax.set_yscale('log', nonposy='clip')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()

        return fig
