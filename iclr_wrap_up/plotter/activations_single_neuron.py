import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from iclr_wrap_up.plotter.base import BasePlotter


def load(run, dataset):
    return SingleNeuronActivityPlotter(run, dataset)


class SingleNeuronActivityPlotter(BasePlotter):
    plotname = 'single_neuron_activations'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def plot(self, measures_summary):
        activations_summary = measures_summary['activations_summary']

        activations_df = pd.DataFrame(activations_summary).transpose()
        all_activations = activations_df['activations']

        num_layers = len(activations_summary[0]['weights_norm'])  # get number of layers indirectly via number of values
        neurons_in_first_layer = all_activations[0][0].shape[1]  # get number of neurons in first layer

        fig = plt.figure()
        gs = gridspec.GridSpec(neurons_in_first_layer * 2, num_layers)

        for layer_number in range(num_layers):
            neurons_in_layer = all_activations[0][layer_number].shape[1]
            hist = [[] for x in range(neurons_in_layer)]
            # grab activations over epoch for every neuron and create histogram
            for epoch, epoch_values in all_activations.items():
                layer_activations = epoch_values[layer_number].transpose()
                for neuron_number in range(neurons_in_layer):
                    histogram_per_neuron = np.histogram(layer_activations[neuron_number], bins=30)[0]
                    hist[neuron_number].append(histogram_per_neuron)

            # plot histogram for every neuron
            for neuron_number in range(neurons_in_layer):
                hist_df = pd.DataFrame(hist[neuron_number])

                # vertical offset for plotting optic
                plotting_offset = (neurons_in_first_layer - neurons_in_layer) / 2
                gs_y_index = int((neuron_number + plotting_offset) * 2)

                ax = fig.add_subplot(gs[gs_y_index:gs_y_index + 2, layer_number])

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
                ax.set_title(f"Layer {layer_number}, Neuron {neuron_number}")

        fig.set_figheight(24)
        fig.set_figwidth(48)
        fig.tight_layout()

        return fig
