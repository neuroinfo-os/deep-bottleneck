import importlib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sacred import Experiment
from sacred.observers import MongoObserver
from iclr_wrap_up.callbacks.loggingreporter import LoggingReporter
import iclr_wrap_up.credentials as credentials

ex = Experiment('sacred_keras_example')

url = f'mongodb://{credentials.MONGODB_ADMINUSERNAME}:{credentials.MONGODB_ADMINPASSWORD}@{credentials.MONGODB_HOST}/?authMechanism=SCRAM-SHA-1'
ex.observers.append(MongoObserver.create(url=url,
                                         db_name=credentials.MONGODB_DBNAME))


@ex.config
def hyperparameters():
    epochs = 10000
    batch_size = 256
    architecture = [10, 7, 2]
    learning_rate = 0.0004
    full_mi = True
    infoplane_measure = 'bin'
    architecture_name = '-'.join(map(str, architecture))
    activation_fn = 'tanh'
    save_dir = 'rawdata/' + activation_fn + '_' + architecture_name
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'
    estimator = 'compute_mi.compute_mi_ib_net'
    callbacks = [('callbacks.earlystopping_manual', [])]
    n_runs = 3


@ex.capture
def load_dataset(dataset):
    module = importlib.import_module(dataset)
    return module.load()


@ex.capture
def load_model(model, architecture, activation_fn, learning_rate, input_size, output_size):
    module = importlib.import_module(model)
    return module.load(architecture, activation_fn, learning_rate, input_size, output_size)


def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:  # Log for all first 20 epochs
        return True
    elif epoch < 100:  # Then for every 5th epoch
        return (epoch % 5) == 0
    elif epoch < 2000:  # Then every 20th
        return (epoch % 20) == 0
    else:  # Then every 100th
        return (epoch % 100) == 0


@ex.capture
def make_callbacks(callbacks, training, test, full_mi, save_dir, batch_size, activation_fn, _run):
    callback_objects = []
    for callback in callbacks:
        callback_object = importlib.import_module(callback[0]).load(*callback[1])
        callback_objects.append(callback_object)
    callback_objects.append(LoggingReporter(trn=training, tst=test, full_mi=full_mi, save_dir=save_dir,
                                 batch_size=batch_size, activation_fn=activation_fn,
                                 do_save_func=do_report))
    return callback_objects


@ex.capture
def load_estimator(estimator, training_data, test_data,
                   full_mi, epochs, architecture_name, activation_fn, infoplane_measure):
    module = importlib.import_module(estimator)
    return module.load(training_data, test_data, epochs, architecture_name, full_mi, activation_fn, infoplane_measure)


@ex.capture
def plot_infoplane(measures, architecture_name, infoplane_measure, epochs, activation_fn):
    os.makedirs('plots/', exist_ok=True)

    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=epochs))
    sm.set_array([])

    fig, ax = plt.subplots()

    for epoch_nr, mi_measures in measures.groupby(level=0):
        color = sm.to_rgba(epoch_nr)

        xmvals = np.array(mi_measures['MI_XM_' + infoplane_measure])
        ymvals = np.array(mi_measures['MI_YM_' + infoplane_measure])

        ax.plot(xmvals, ymvals, color=color, alpha=0.1, zorder=1)
        ax.scatter(xmvals, ymvals, s=20, facecolors=color, edgecolor='none', zorder=2)

    ax.set(xlim=[0, 12], ylim=[0, 1], xlabel='I(X;M)', ylabel='I(Y;M)')

    plt.colorbar(sm, label='Epoch')

    filename = f'plots/infoplane_{activation_fn}_{architecture_name}_{infoplane_measure}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    return filename


@ex.capture
def plot_snr(architecture_name, activation_fn, architecture):
    cur_dir = f'rawdata/{activation_fn}_{architecture_name}'

    epochs = []
    means = []
    stds = []
    wnorms = []

    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            continue

        with open(os.path.join(cur_dir, epochfile), 'rb') as fh:
            d = pickle.load(fh)

        epoch = d['epoch']
        epochs.append(epoch)
        wnorms.append(d['data']['weights_norm'])
        means.append(d['data']['gradmean'])
        stds.append(d['data']['gradstd'])

    wnorms, means, stds = map(np.array, [wnorms, means, stds])
    plot_layers = range(len(architecture) + 1)  # +1 for the last output layer.

    fig, axes = plt.subplots(ncols=len(plot_layers), figsize=(12, 5))

    for lndx, layerid in enumerate(plot_layers):
        axes[lndx].plot(epochs, means[:, layerid], 'b', label="Mean")
        axes[lndx].plot(epochs, stds[:, layerid], 'orange', label="Std")
        axes[lndx].plot(epochs, means[:, layerid] / stds[:, layerid], 'red', label="SNR")
        axes[lndx].plot(epochs, wnorms[:, layerid], 'g', label="||W||")

        axes[lndx].set_title(f'Layer {layerid}')
        axes[lndx].set_xlabel('Epoch')
        axes[lndx].set_xscale("log", nonposx='clip')
        axes[lndx].set_yscale("log", nonposy='clip')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    filename = f'plots/snr_{activation_fn}_{architecture_name}.png'
    fig.savefig(filename, bbox_inches='tight', dpi=600)
    return filename


@ex.automain
def conduct(epochs, batch_size, n_runs, _run):
    training, test = load_dataset()

    measures_all_runs = []
    for run_id in range(n_runs):
        model = load_model(input_size=training.X.shape[1], output_size=training.nb_classes)
        callbacks = make_callbacks(training=training, test=test)
        model.fit(x=training.X, y=training.Y,
                  verbose=2,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(test.X, test.Y),
                  callbacks=callbacks)

        estimator = load_estimator(training_data=training, test_data=test)
        measures = estimator.compute_mi()
        measures_all_runs.append(measures)

    # transform list of measurements into DataFrame with hierarchical index
    measures_all_runs = pd.concat(measures_all_runs)
    measures_all_runs = measures_all_runs.fillna(0)
    # compute mean of information measures over all runs
    mi_mean_over_runs = measures_all_runs.groupby(['epoch', 'layer']).mean()


    # plot the infoplane for average MI estimates
    filename = plot_infoplane(measures=mi_mean_over_runs)
    _run.add_artifact(filename, name='infoplane_plot')
    # TODO think about whether plotting snr ratio averaged over multiple runs does make sense
    filename = plot_snr()
    _run.add_artifact(filename, name='snr_plot')

