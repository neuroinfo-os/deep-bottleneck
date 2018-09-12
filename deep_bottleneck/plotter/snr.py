import matplotlib.pyplot as plt
import numpy as np

from deep_bottleneck.plotter.base import BasePlotter


def load(run, dataset):
    return SignalToNoiseRatioPlotter(run, dataset)


# TODO think about whether plotting snr ratio averaged over multiple runs does make sense

class SignalToNoiseRatioPlotter(BasePlotter):
    plotname = 'snr'
    file_ext = 'png'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def plot(self, measures_summary):

        activations_summary = measures_summary['activations_summary']
        num_layers = len(
            activations_summary["0"]['weights_norm'])  # get number of layers indirectly via number of values

        epochs = []
        means = []
        stds = []
        wnorms = []

        epochs_in_activation_summary = [int(epoch) for epoch in activations_summary]
        epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))

        for epoch in epochs_in_activation_summary:
            epochs.append(epoch)
            epoch_values = activations_summary[str(epoch)]
            wnorms.append(epoch_values['weights_norm'])
            means.append(epoch_values['gradmean'])
            stds.append(epoch_values['gradstd'])

        wnorms, means, stds = map(np.array, [wnorms, means, stds])
        plot_layers = range(num_layers)

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
