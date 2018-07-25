import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iclr_wrap_up import utils

from iclr_wrap_up.plotter.base import BasePlotter


def load(run, dataset):
    return ActivityPlotter(run, dataset)


class ActivityPlotter(BasePlotter):
    plotname = 'activations'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run


    def plot(self, measures_summary):
        activations_summary = measures_summary['activations_summary']
        num_layers = len(activations_summary[0]['weights_norm'])  # get number of layers indirectly via number of values

        activations_df = pd.DataFrame(activations_summary).transpose()
        all_activations = activations_df['activations']

        fig = plt.figure()

        for layer in range(num_layers):
            ax = fig.add_subplot(num_layers, 1, layer + 1)

            min, max = utils.get_min_max(all_activations, layer_number=layer)
            bins = np.linspace(min, max, 30)

            hist = []
            for epoch, activations in all_activations.items():
                hist.append(np.histogram(activations[layer], bins=bins)[0])

            hist_df = pd.DataFrame(hist)

            ax.set_ylabel("bins")
            yticks = np.arange(0, hist_df.shape[1], 5)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)

            ax.set_xlabel("epoch")
            xticks = np.arange(0, hist_df.shape[0], 5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(all_activations.index[xticks], rotation=90)

            activity_map = ax.imshow(hist_df.transpose(), cmap="viridis", interpolation='nearest')
            counts_colorbar = fig.colorbar(activity_map)
            counts_colorbar.set_label("Absolute frequency")
            ax.set_title(f"Layer {layer}")

        fig.set_figheight(12)
        fig.set_figwidth(16)
        fig.tight_layout()

        return fig
