import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deep_bottleneck import utils
from deep_bottleneck.plotter.base import BasePlotter


def load(run, dataset):
    return ActivityPlotter(run, dataset)


class ActivityPlotter(BasePlotter):
    plotname = 'activations'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def plot(self, measures_summary):
        activations_summary = measures_summary['activations_summary']
        num_layers = len(activations_summary["0"]['weights_norm'])  # get number of layers indirectly via number of values

        fig = plt.figure()

        for layer in range(num_layers):
            ax = fig.add_subplot(num_layers, 1, layer + 1)

            min_activations, max_activations = utils.get_min_max(activations_summary, layer_number=layer)
            bins = np.linspace(min_activations, max_activations, 30)

            hist = []
            epochs_in_activation_summary = [int(epoch) for epoch in activations_summary]
            epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))

            for epoch in epochs_in_activation_summary:
                hist.append(np.histogram(activations_summary[f'{epoch}/activations/{layer}'], bins=bins)[0])

            hist_df = pd.DataFrame(hist)

            ax.set_ylabel("bins")
            yticks = np.arange(0, hist_df.shape[1], 5)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)

            ax.set_xlabel("epoch")
            xticks = np.arange(0, hist_df.shape[0], 5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(epochs_in_activation_summary[xticks], rotation=90)

            activity_map = ax.imshow(hist_df.transpose(), cmap="viridis", interpolation='nearest')
            counts_colorbar = fig.colorbar(activity_map)
            counts_colorbar.set_label("Absolute frequency")
            ax.set_title(f"Layer {layer}")

        fig.set_figheight(12)
        fig.set_figwidth(16)
        fig.tight_layout()

        return fig
