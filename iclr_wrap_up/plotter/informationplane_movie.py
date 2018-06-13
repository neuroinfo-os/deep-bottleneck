import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from iclr_wrap_up.plotter.base import BasePlotter


def load(run, dataset):
    return InformationPlaneMoviePlotter(run, dataset)


class InformationPlaneMoviePlotter(BasePlotter):
    """Plot the infoplane movie for several runs of the same network."""
    plotname = 'infoplane_movie'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def plot(self, measures_summary):

        measures = measures_summary['measures_all_runs']

        os.makedirs('plots/', exist_ok=True)

        fig, ax = plt.subplots()
        if self.dataset == 'datasets.mnist' or self.dataset == 'datasets.fashion_mnist':
            ax.set(xlim=[0, 14], ylim=[0, 3.5])
        else:
            ax.set(xlim=[0, 12], ylim=[0, 1])


        scatter = ax.scatter([], [], s=20, edgecolor='none', zorder=2)

        num_layers = measures.index.get_level_values(1).nunique()
        layers_colors = np.random.rand(num_layers)

        writer = FFMpegWriter(fps=10)

        with writer.saving(fig, "plots/writer_test.mp4", 600):
            for epoch_number, mi_epoch in measures.groupby(level=0):
                # Drop outer index level corresponding to the epoch.
                mi_epoch.index = mi_epoch.index.droplevel()

                xmvals = mi_epoch['MI_XM']
                ymvals = mi_epoch['MI_YM']

                points = np.array([xmvals, ymvals]).transpose()
                colors = layers_colors[mi_epoch.index]

                scatter.set_offsets(points)
                scatter.set_array(colors)

                writer.grab_frame()

        ax.set(xlabel='I(X;M)', ylabel='I(Y;M)')


        return fig