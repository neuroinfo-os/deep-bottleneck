import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


from iclr_wrap_up.plotter.base import BasePlotter
from iclr_wrap_up import utils


def load(run, dataset):
    return SingleNeuronActivityPlotter(run, dataset)


class SingleNeuronActivityPlotter(BasePlotter):
    plotname = 'single_neuron_activations'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def _grab_activations(self, measures_summary):
        activations_summary = measures_summary['activations_summary']
        return activations_summary

    def _get_number_of_layers(self, all_activations):
        return len(all_activations["0/activations"])

    def _get_number_of_neurons_in_layer(self, all_activations, layer):
        return all_activations["0/activations"][str(layer)].shape[1]

    def _create_histogram(self, all_activations, layer_number):
        neurons_in_layer = self._get_number_of_neurons_in_layer(all_activations, layer_number)
        hist = [[] for x in range(neurons_in_layer)]
        epochs_in_activation_summary = [int(epoch) for epoch in all_activations]
        epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))
        for epoch in epochs_in_activation_summary:
            activations = all_activations[str(epoch)]
            layer_activations = activations[f'activations/{layer_number}']
            layer_activations = np.asarray(layer_activations)
            layer_activations = layer_activations.transpose()
            for neuron_number in range(neurons_in_layer):
                activations_min, activations_max = utils.get_min_max(all_activations, layer_number, neuron_number)
                bins = np.linspace(activations_min, activations_max, 30)

                histogram_per_neuron, _ = np.histogram(layer_activations[neuron_number], bins=bins)
                hist[neuron_number].append(histogram_per_neuron)
        return hist

    def plot(self, measures_summary):

        all_activations = self._grab_activations(measures_summary)
        neurons_in_first_layer = self._get_number_of_neurons_in_layer(all_activations, 0)
        num_layers = self._get_number_of_layers(all_activations)

        fig = plt.figure()
        gs = gridspec.GridSpec(neurons_in_first_layer * 2, num_layers)

        for layer_number in range(num_layers):

            neurons_in_layer = self._get_number_of_neurons_in_layer(all_activations, layer_number)
            hist = self._create_histogram(all_activations, layer_number)

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
                epochs_in_activation_summary = [int(epoch) for epoch in all_activations]
                epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))
                ax.set_xticks(xticks)
                ax.set_xticklabels(epochs_in_activation_summary[xticks], rotation=90)

                activity_map = ax.imshow(hist_df.transpose(), cmap="viridis", interpolation='nearest')
                counts_colorbar = fig.colorbar(activity_map)
                counts_colorbar.set_label("Absolute frequency")
                ax.set_title(f"Layer {layer_number}, Neuron {neuron_number}")

        fig.set_figheight(24)
        fig.set_figwidth(48)
        fig.tight_layout()

        return fig
