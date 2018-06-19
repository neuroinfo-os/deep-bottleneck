import os
import matplotlib.pyplot as plt
import numpy as np

from iclr_wrap_up.plotter.base import BasePlotter


def load(run, dataset, epochs):
    return InformationPlanePlotter(run, dataset, epochs)


class InformationPlanePlotter(BasePlotter):
    """Plot the infoplane for average MI estimates."""
    plotname = 'infoplane'

    def __init__(self, run, dataset, epochs):
        self.dataset = dataset
        self.epochs = epochs
        self.run = run

    def plot(self, measures_summary):

        measures = measures_summary['mi_mean_over_runs']

        os.makedirs('plots/', exist_ok=True)

        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=self.epochs))
        sm.set_array([])

        fig, ax = plt.subplots()

        for epoch_nr, mi_measures in measures.groupby(level=0):
            color = sm.to_rgba(epoch_nr)

            xmvals = np.array(mi_measures['MI_XM'])
            ymvals = np.array(mi_measures['MI_YM'])

            ax.plot(xmvals, ymvals, color=color, alpha=0.1, zorder=1)
            ax.scatter(xmvals, ymvals, s=20, facecolors=color, edgecolor='none', zorder=2)

        ax.set(xlabel='I(X;M)', ylabel='I(Y;M)')

        if self.dataset == 'datasets.mnist' or self.dataset == 'datasets.fashion_mnist':
            ax.set(xlim=[0, 14], ylim=[0, 3.5])
        else:
            ax.set(xlim=[0, 12], ylim=[0, 1])

        fig.colorbar(sm, label='Epoch')

        return fig