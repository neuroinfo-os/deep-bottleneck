import os
import matplotlib.pyplot as plt
import numpy as np


def load(run, architecture_name, infoplane_measure, epochs, activation_fn):
    return InformationPlanePlotter(run, architecture_name, infoplane_measure, epochs, activation_fn)

class InformationPlanePlotter:
    '''
    Plot the infoplane for average MI estimates.
    '''


    def __init__(self, run, architecture_name, infoplane_measure, epochs, activation_fn):
        self.architecture_name = architecture_name
        self.infoplane_measure = infoplane_measure
        self.epochs = epochs
        self.activation_fn = activation_fn
        self.run = run


    def generate_plot(self, measures_summary):

        measures = measures_summary['mi_mean_over_runs']

        os.makedirs('plots/', exist_ok=True)

        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=self.epochs))
        sm.set_array([])

        fig, ax = plt.subplots()

        for epoch_nr, mi_measures in measures.groupby(level=0):
            color = sm.to_rgba(epoch_nr)

            xmvals = np.array(mi_measures['MI_XM_' + self.infoplane_measure])
            ymvals = np.array(mi_measures['MI_YM_' + self.infoplane_measure])

            ax.plot(xmvals, ymvals, color=color, alpha=0.1, zorder=1)
            ax.scatter(xmvals, ymvals, s=20, facecolors=color, edgecolor='none', zorder=2)

        ax.set(xlim=[0, 12], ylim=[0, 1], xlabel='I(X;M)', ylabel='I(Y;M)')

        plt.colorbar(sm, label='Epoch')

        filename = f'plots/infoplane_{self.activation_fn}_{self.architecture_name}_{self.infoplane_measure}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=600)

        self.run.add_artifact(filename, name='infoplane_plot')
