import importlib
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.gridspec as gridspec

from sacred import Experiment
from sacred.observers import MongoObserver
from iclr_wrap_up.callbacks.loggingreporter import LoggingReporter
from sacred.observers import FileStorageObserver
from labwatch.assistant import LabAssistant
from labwatch.optimizers.random_search import RandomSearch
from labwatch.hyperparameters import UniformNumber, Categorical


ex = Experiment('sacred_keras_example')
a = LabAssistant(ex, "labwatch_example", optimizer=RandomSearch)
ex.observers.append(MongoObserver.create(url='mongodb://127.0.0.1:27017',
                                         db_name='dneck_test'))


@ex.config
def hyperparameters():
    epochs = 10
    batch_size = 256
    architecture = [4, 3]
    learning_rate = 0.0004
    full_mi = True
    infoplane_measure = 'upper'
    architecture_name= '-'.join(map(str, architecture))
    activation_fn = 'relu'
    save_dir = 'rawdata/' + activation_fn + '_' + architecture_name
    infoplane_measure = 'upper'
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'
    estimator = 'compute_mi.compute_mi_ib_net'

@a.search_space
def search_space():
    activation_fn = Categorical({'relu', 'tanh'})
    batch_size = UniformNumber(lower=128,
                               upper=512,
                               default=256,
                               type=int,
                               log_scale=True)


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
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5) == 0
    elif epoch < 2000:    # Then every 10th
        return (epoch % 20) == 0
    else:                # Then every 100th
        return (epoch % 100) == 0


@ex.capture
def make_callbacks(training, test, full_mi, save_dir, batch_size, activation_fn, _run):
    callbacks = [LoggingReporter(trn=training, tst=test, full_mi=full_mi, save_dir=save_dir,
                                 batch_size=batch_size, activation_fn=activation_fn,
                                 do_save_func=do_report)]
    return callbacks

@ex.capture
def load_estimator(estimator, training_data, test_data,
                   full_mi, epochs, architecture_name, activation_fn, infoplane_measure):
    module = importlib.import_module(estimator)
    return module.load(training_data, test_data, epochs, architecture_name, full_mi, activation_fn, infoplane_measure)

@ex.capture
def plot_infoplane(measures, plot_layers, architecture_name, infoplane_measure, epochs):
    DIR_TEMPLATE = '%%s_%s' % architecture_name
    os.makedirs('plots/', exist_ok=True)

    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=epochs))
    sm._A = []

    fig = plt.figure(figsize=(10, 5))
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        plt.subplot(1, 2, actndx + 1)
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_' + infoplane_measure])[plot_layers]
            ymvals = np.array(vals[epoch]['MI_YM_' + infoplane_measure])[plot_layers]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in plot_layers], edgecolor='none', zorder=2)

        plt.ylim([0, 1])
        plt.xlim([0, 12])
        #     plt.ylim([0, 3.5])
        #     plt.xlim([0, 14])
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()

    plt.savefig('plots/' + DIR_TEMPLATE % ('infoplane_' + architecture_name),
                bbox_inches='tight')

    return actndx

@ex.capture
def plot_snr(measures, plot_layers, actndx, architecture_name):
    DIR_TEMPLATE = '%%s_%s' % architecture_name
    plt.figure(figsize=(12, 5))

    gs = gridspec.GridSpec(len(measures), len(plot_layers))
    for activation in measures.keys():
        cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
        if not os.path.exists(cur_dir):
            continue

        epochs = []
        means = []
        stds = []
        wnorms = []
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue

            with open(cur_dir + "/" + epochfile, 'rb') as f:
                d = pickle.load(f)

            epoch = d['epoch']
            epochs.append(epoch)
            wnorms.append(d['data']['weights_norm'])
            means.append(d['data']['gradmean'])
            stds.append(d['data']['gradstd'])

        wnorms, means, stds = map(np.array, [wnorms, means, stds])
        for lndx, layerid in enumerate(plot_layers):
            plt.subplot(gs[actndx, lndx])
            plt.plot(epochs, means[:, layerid], 'b', label="Mean")
            plt.plot(epochs, stds[:, layerid], 'orange', label="Std")
            plt.plot(epochs, means[:, layerid] / stds[:, layerid], 'red', label="SNR")
            plt.plot(epochs, wnorms[:, layerid], 'g', label="||W||")

            plt.title('Layer %d' % layerid)
            plt.xlabel('Epoch')
            plt.gca().set_xscale("log", nonposx='clip')
            plt.gca().set_yscale("log", nonposy='clip')

    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
    plt.tight_layout()
    plt.savefig('plots/' + DIR_TEMPLATE % ('snr_' + architecture_name), bbox_inches='tight')


@ex.automain
def conduct(epochs, batch_size):
    training, test = load_dataset()
    model = load_model(input_size=training.X.shape[1], output_size=training.nb_classes)
    callbacks = make_callbacks(training=training, test=test)
    model.fit(x=training.X, y=training.Y,
                  verbose=2,
                  batch_size=batch_size,
                  epochs=epochs,
                  # validation_data=(tst.X, tst.Y),
                  callbacks=callbacks)
    estimator = load_estimator(training_data=training, test_data=test)
    measures, plot_layers = estimator.compute_mi()
    actndx = plot_infoplane(measures=measures, plot_layers=plot_layers)
    plot_snr(measures=measures, plot_layers=plot_layers, actndx=actndx)

