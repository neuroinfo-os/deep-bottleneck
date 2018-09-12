import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from deep_bottleneck.plotter.base import BasePlotter


def load(run, dataset):
    return InformationPlaneMoviePlotter(run, dataset)


class InformationPlaneMoviePlotter(BasePlotter):
    """Plot the infoplane movie for several runs of the same network."""
    plotname = 'infoplane_movie'
    file_ext = 'mp4'


    num_layers = None
    total_number_of_epochs = None
    epoch_indexes = None
    layers_colors = None

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def generate(self, measures_summary, suffix):
        self.filename = self.make_filename(suffix)
        self.plot(measures_summary)
        self.run.add_artifact(self.filename, name=self.plotname)

    def setup_infoplane_subplot(self, ax_infoplane):
        if self.dataset == 'datasets.mnist' or self.dataset == 'datasets.fashion_mnist':
            ax_infoplane.set(xlim=[0, 14], ylim=[0, 3.5])
        else:
            ax_infoplane.set(xlim=[0, 12], ylim=[0, 1])

        ax_infoplane.set(xlabel='I(X;M)', ylabel='I(Y;M)')

        scatter = ax_infoplane.scatter([], [], s=20, edgecolor='none')
        text = ax_infoplane.text(0, 1.05, "", fontsize=12)

        return scatter, text

    def fill_infoplane_subplot(self, ax_infoplane, mi_epoch):
        xmvals = mi_epoch['MI_XM']
        ymvals = mi_epoch['MI_YM']

        points = np.array([xmvals, ymvals]).transpose()
        colors = self.layers_colors[mi_epoch.index]

        ax_infoplane.set_offsets(points)
        ax_infoplane.set_array(colors)

        return ax_infoplane

    def setup_accuracy_subplot(self, ax_accuracy):
        [acc_line] = ax_accuracy.plot([], [], 'b', label="training accuracy")
        [val_acc_line] = ax_accuracy.plot([], [], 'g', label="test accuracy")

        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.set_xlim(0, self.total_number_of_epochs)
        if self.total_number_of_epochs > 20:
            xticks_positions = range(0, self.total_number_of_epochs, int(self.total_number_of_epochs / 20))
            ax_accuracy.set_xticks(xticks_positions)
            ax_accuracy.set_xticklabels(self.epoch_indexes[xticks_positions], rotation=90)

        handles, labels = ax_accuracy.get_legend_handles_labels()
        ax_accuracy.legend(handles, labels, loc=4)

        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy')

        return acc_line, val_acc_line

    def fill_accuracy_subplot(self, acc_line, val_acc_line, activations_summary, epoch_number, acc, val_acc):
        epoch_accuracy = np.asarray(activations_summary[f'{epoch_number}/accuracy/']['training'])
        epoch_val_accuracy = np.asarray(activations_summary[f'{epoch_number}/accuracy/']['test'])

        acc.append(epoch_accuracy)
        val_acc.append(epoch_val_accuracy)

        xs = range(len(acc))
        acc_line.set_data(xs, acc)
        val_acc_line.set_data(xs, val_acc)

        return acc, val_acc, acc_line, val_acc_line

    def get_specifications(self, measures):
        self.num_layers = measures.index.get_level_values(1).nunique()
        self.layers_colors = np.linspace(0, 1, self.num_layers)
        self.epoch_indexes = measures.index.get_level_values('epoch').unique()
        self.total_number_of_epochs = len(self.epoch_indexes)

    def plot(self, measures_summary):
        os.makedirs('plots/', exist_ok=True)

        measures = measures_summary['measures_all_runs']
        activations_summary = measures_summary['activations_summary']
        self.get_specifications(measures)

        plt.set_cmap("hsv")
        fig, (ax_infoplane, ax_accuracy) = plt.subplots(2, 1, figsize=(6, 9),
                                                        gridspec_kw={'height_ratios': [2, 1]})

        acc = []
        val_acc = []

        scatter, text = self.setup_infoplane_subplot(ax_infoplane)
        acc_line, val_acc_line = self.setup_accuracy_subplot(ax_accuracy)

        writer = FFMpegWriter(fps=7)
        with writer.saving(fig, self.filename, 600):
            for epoch_number, mi_epoch in measures.groupby(level=0):
                # Drop outer index level corresponding to the epoch.
                mi_epoch.index = mi_epoch.index.droplevel()

                scatter = self.fill_infoplane_subplot(scatter, mi_epoch)
                text.set_text(f'Epoch: {epoch_number}')

                acc, val_acc, acc_line, val_acc_line = self.fill_accuracy_subplot(acc_line, val_acc_line,
                                                          activations_summary, epoch_number,
                                                          acc, val_acc)

                writer.grab_frame()
