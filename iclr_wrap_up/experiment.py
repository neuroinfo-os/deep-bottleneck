import importlib
import pandas as pd
import datetime

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.callbacks import TensorBoard
from iclr_wrap_up.callbacks.loggingreporter import LoggingReporter
from iclr_wrap_up.callbacks.metrics_logger import MetricsLogger
from iclr_wrap_up.callbacks.activityprojector import ActivityProjector
import matplotlib
matplotlib.use('agg')

import iclr_wrap_up.credentials as credentials

ex = Experiment('sacred_keras_example')

url = f'mongodb://{credentials.MONGODB_ADMINUSERNAME}:{credentials.MONGODB_ADMINPASSWORD}@{credentials.MONGODB_HOST}/?authMechanism=SCRAM-SHA-1'
ex.observers.append(MongoObserver.create(url=url,
                                         db_name=credentials.MONGODB_DBNAME))


@ex.config
def hyperparameters():
    epochs = 10
    batch_size = 256
    architecture = [10, 7, 5, 4, 3]
    learning_rate = 0.0004
    full_mi = False
    infoplane_measure = 'upper'
    architecture_name = '-'.join(map(str, architecture))
    activation_fn = 'tanh'
    save_dir = 'rawdata/' + activation_fn + '_' + architecture_name
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'
    estimator = 'compute_mi.compute_mi_ib_net'
    callbacks = []
    plotters = [('plotter.informationplane', [epochs]),
               ('plotter.snr', [architecture]),
               ('plotter.informationplane_movie', [])]
    n_runs = 5


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
    if epoch < 50:  # Log for all first 50 epochs
        return True
    elif epoch < 100:  # Then for every 2th epoch
        return (epoch % 2) == 0
    elif epoch < 500:  # Then for every 5th epoch
        return (epoch % 5) == 0
    elif epoch < 2000:  # Then every 20th
        return (epoch % 20) == 0
    else:  # Then every 100th
        return (epoch % 100) == 0

@ex.capture
def make_plotters(plotters, _run, dataset):

    plotter_objects = []
    for plotter in plotters:
        plotter_object = importlib.import_module(plotter[0]).load(_run, dataset, *plotter[1])
        plotter_objects.append(plotter_object)

    return plotter_objects

@ex.capture
def generate_plots(plotter_objects, measures_summary):

    for plotter in plotter_objects:
        plotter.generate(measures_summary)


@ex.capture
def make_callbacks(callbacks, training, test, full_mi, save_dir, batch_size, activation_fn, _run):
    datestr = str(datetime.datetime.now()).split(sep='.')[0]
    datestr = datestr.replace(':', '-')
    datestr = datestr.replace(' ', '_')

    callback_objects = []
    # The logging reporter needs to be at position 0 to access the correct one for the further processing.
    callback_objects.append(LoggingReporter(trn=training, tst=test, full_mi=full_mi,
                                            batch_size=batch_size, activation_fn=activation_fn,
                                            do_save_func=do_report))
    for callback in callbacks:
        callback_object = importlib.import_module(callback[0]).load(*callback[1])
        callback_objects.append(callback_object)
    callback_objects.append(MetricsLogger(_run))
    callback_objects.append(TensorBoard(log_dir=f'./logs/{datestr}', histogram_freq=10))
    callback_objects.append(ActivityProjector(log_dir=f'./logs/{datestr}', train=training, test=test,
                                   embeddings_freq=10))

    return callback_objects


@ex.capture
def load_estimator(estimator, training_data, test_data,
                   full_mi, epochs, architecture_name, activation_fn, infoplane_measure):
    module = importlib.import_module(estimator)
    return module.load(training_data, test_data, epochs, architecture_name, full_mi, activation_fn, infoplane_measure)


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

        print('fit successful')

        # Getting the current activations_summary from the logging_callback.
        activations_summary = callbacks[0].activations_summary

        estimator = load_estimator(training_data=training, test_data=test)
        measures = estimator.compute_mi(activations_summary=activations_summary)
        measures['run'] = run_id
        measures_all_runs.append(measures)

        # Clear the current Session to free current layer and model definition.
        # This would otherwise be kept in memory. It is not needed as every run
        # redefines the model.
        K.clear_session()

    # Transform list of measurements into DataFrame with hierarchical index.
    measures_all_runs = pd.concat(measures_all_runs)
    measures_all_runs = measures_all_runs.fillna(0)
    # compute mean of information measures over all runs
    mi_mean_over_runs = measures_all_runs.groupby(['epoch', 'layer']).mean()

    measures_summary = {'measures_all_runs': measures_all_runs,
                        'mi_mean_over_runs': mi_mean_over_runs,
                        'activations_summary': activations_summary}

    plotter_objects = make_plotters()
    generate_plots(plotter_objects, measures_summary)