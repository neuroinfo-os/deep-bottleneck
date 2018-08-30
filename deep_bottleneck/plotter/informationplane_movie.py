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
    filename = f'plots/{plotname}.mp4'

    def __init__(self, run, dataset):
        self.dataset = dataset
        self.run = run

    def generate(self, measures_summary):
        self.plot(measures_summary)
        self.run.add_artifact(self.filename, name=self.plotname)

    def plot(self, measures_summary):

        measures = measures_summary['measures_all_runs']
        activations_summary = measures_summary['activations_summary']

        os.makedirs('plots/', exist_ok=True)

        plt.set_cmap("hsv")
        fig, (ax_infoplane, ax_accuracy) = plt.subplots(2,1, figsize=(6, 9),
                                                        gridspec_kw={'height_ratios': [2, 1]})

        if self.dataset == 'datasets.mnist' or self.dataset == 'datasets.fashion_mnist':
            ax_infoplane.set(xlim=[0, 14], ylim=[0, 3.5])
        else:
            ax_infoplane.set(xlim=[0, 12], ylim=[0, 1])

            ax_infoplane.set(xlabel='I(X;M)', ylabel='I(Y;M)')

        scatter = ax_infoplane.scatter([], [], s=20, edgecolor='none')
        text = ax_infoplane.text(0, 1.05, "", fontsize=12)

        num_layers = measures.index.get_level_values(1).nunique()
        layers_colors = np.linspace(0, 1, num_layers)

        acc = []
        val_acc = []
        acc_line,  = ax_accuracy.plot([], [], 'b', label="training accuracy")
        val_acc_line, = ax_accuracy.plot([], [], 'g', label="validation accuracy")

        epoch_indexes = measures.index.get_level_values('epoch').unique()
        total_number_of_epochs = len(epoch_indexes)

        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.set_xlim(0, total_number_of_epochs)

        xticks_positions = range(0, total_number_of_epochs, int(total_number_of_epochs/20))
        ax_accuracy.set_xticks(xticks_positions)
        ax_accuracy.set_xticklabels(epoch_indexes[xticks_positions], rotation=90)

        handles, labels = ax_accuracy.get_legend_handles_labels()
        ax_accuracy.legend(handles, labels, loc=4)
        
        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy')

        writer = FFMpegWriter(fps=7)
        with writer.saving(fig, self.filename, 600):
            for epoch_number, mi_epoch in measures.groupby(level=0):
                # Drop outer index level corresponding to the epoch.
                mi_epoch.index = mi_epoch.index.droplevel()

                text.set_text(f"Epoch: {epoch_number}")

                # infoplane subplot
                xmvals = mi_epoch['MI_XM']
                ymvals = mi_epoch['MI_YM']

                points = np.array([xmvals, ymvals]).transpose()
                colors = layers_colors[mi_epoch.index]

                scatter.set_offsets(points)
                scatter.set_array(colors)

                # accuracy subplot
                epoch_accuracy = np.asarray(activations_summary[f'{epoch_number}/accuracy/']['training'])
                epoch_val_accuracy = np.asarray(activations_summary[f'{epoch_number}/accuracy/']['validation'])

                acc.append(epoch_accuracy)
                val_acc.append(epoch_val_accuracy)

                xs = range(len(acc))
                acc_line.set_data(xs, acc)
                val_acc_line.set_data(xs, val_acc)

                writer.grab_frame()

