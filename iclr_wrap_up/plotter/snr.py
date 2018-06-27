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

        for epoch, epoch_values in activations_summary.items():
            epochs.append(epoch)
            wnorms.append(epoch_values['weights_norm'])
            means.append(epoch_values['gradmean'])
            stds.append(epoch_values['gradstd'])

        wnorms, means, stds = map(np.array, [wnorms, means, stds])
        plot_layers = range(len(self.architecture) + 1)  # +1 for the last output layer.

        fig, axes = plt.subplots(ncols=len(plot_layers), figsize=(12, 5))

        for lndx, layerid in enumerate(plot_layers):
            axes[lndx].plot(epochs, means[:, layerid], 'b', label='Mean')
            axes[lndx].plot(epochs, stds[:, layerid], 'orange', label='Std')
            axes[lndx].plot(epochs, means[:, layerid] / stds[:, layerid], 'red', label='SNR')
            axes[lndx].plot(epochs, wnorms[:, layerid], 'g', label='||W||')

            axes[lndx].set_title(f'Layer {layerid}')
            axes[lndx].set_xlabel('Epoch')
            axes[lndx].set_xscale('log', nonposx='clip')
            axes[lndx].set_yscale('log', nonposy='clip')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()

        return fig
