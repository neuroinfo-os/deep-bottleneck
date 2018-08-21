import numpy as np
import importlib
import sklearn.manifold

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from deep_bottleneck import utils
from deep_bottleneck.plotter.base import BasePlotter


def load(run, dataset):
    return MultidimensionalScaling(run, dataset)


class MultidimensionalScaling(BasePlotter):
    plotname = 'multidimensional_scaling'
    filename = f'plots/{plotname}.mp4'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def generate(self, measures_summary):
        self.plot(measures_summary)
        self.run.add_artifact(self.filename, name=self.plotname)

    def plot(self, measures_summary):
        activations_summary = measures_summary['activations_summary']

        # TODO make dataset available such that it doesnt have to be reloaded everywhere
        _, test = importlib.import_module(self.dataset).load()
        #original_dataset = utils.construct_full_dataset(training, test)
        labels_colors = test.y

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [], cmap='viridis', marker='.')

        writer = FFMpegWriter(fps=3)

        with writer.saving(fig, self.filename, 600):
            layer = 4
            epochs_in_activation_summary = [int(epoch) for epoch in activations_summary]
            epochs_in_activation_summary = np.asarray(sorted(epochs_in_activation_summary))

            for epoch in epochs_in_activation_summary:
                activations = np.asarray(activations_summary[f'{epoch}/activations/{layer}'], dtype=np.float64)

                print(f'Doing multidimensional scaling for epoch {epoch}')
                mds = sklearn.manifold.MDS(2, max_iter=50, n_init=1)
                transformed_activations = mds.fit_transform(activations)

                scatter.set_offsets(transformed_activations)
                scatter.set_array(labels_colors)

                ax.set_title(f"Layer {layer}, Epoch {epoch}")

                min_x = np.min(transformed_activations)
                max_x = np.max(transformed_activations)
                min_y = np.min(transformed_activations)
                max_y = np.max(transformed_activations)

                ax.set_xlim([min_x, max_x])
                ax.set_ylim([min_y, max_y])

                writer.grab_frame()
