import importlib
import pandas as pd
import datetime
import h5py
import os

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.callbacks import TensorBoard
from deep_bottleneck.callbacks.loggingreporter import LoggingReporter
from deep_bottleneck.callbacks.metrics_logger import MetricsLogger
from deep_bottleneck.callbacks.activityprojector import ActivityProjector
import matplotlib

matplotlib.use('agg')

import deep_bottleneck.credentials as credentials

ex = Experiment('sacred_keras_example')

ex.observers.append(MongoObserver.create(url=credentials.MONGODB_URI,
                                         db_name=credentials.MONGODB_DBNAME))


@ex.config
def hyperparams():
    # For downwards compatibility
    dataset = None
    max_norm_weights = False
    if dataset == 'dataset.mnist':
        plotters = [('plotter.informationplane', []),
                    ('plotter.snr', []),
                    ('plotter.informationplane_movie', []),
                    ('plotter.activations', [])
                    ]
    else:
        plotters = [('plotter.informationplane', []),
                    ('plotter.snr', []),
                    ('plotter.informationplane_movie', []),
                    ('plotter.activations', []),
                    ('plotter.activations_single_neuron', [])
                    ]


ex.add_config('configs/basic.json')


@ex.capture
def load_dataset(dataset):
    module = importlib.import_module(dataset)
    return module.load()


@ex.capture
def load_model(model, architecture, activation_fn, optimizer, learning_rate, input_size, output_size, max_norm_weights):
    module = importlib.import_module(model)
    return module.load(architecture, activation_fn, optimizer, learning_rate, input_size, output_size, max_norm_weights)


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
def make_callbacks(callbacks, data, calculate_mi_for, batch_size, activation_fn, _run, file_all_activations):
    datestr = str(datetime.datetime.now()).split(sep='.')[0]
    datestr = datestr.replace(':', '-')
    datestr = datestr.replace(' ', '_')

    callback_objects = []
    # The logging reporter needs to be at position 0 to access the correct one for the further processing.
    callback_objects.append(LoggingReporter(data, calculate_mi_for=calculate_mi_for,
                                            batch_size=batch_size, activation_fn=activation_fn,
                                            do_save_func=do_report, file_all_activations=file_all_activations))
    for callback in callbacks:
        callback_object = importlib.import_module(callback[0]).load(*callback[1])
        callback_objects.append(callback_object)
    callback_objects.append(MetricsLogger(_run))
    callback_objects.append(TensorBoard(log_dir=f'./logs/{datestr}', histogram_freq=10))
    callback_objects.append(ActivityProjector(data.test,
                                              log_dir=f'./logs/{datestr}',
                                              embeddings_freq=10))

    return callback_objects


@ex.capture
def load_estimator(estimator, discretization_range, data, calculate_mi_for, architecture):
    module = importlib.import_module(estimator)
    return module.load(discretization_range, data, architecture, calculate_mi_for)


@ex.automain
def conduct(epochs, batch_size, n_runs, _run):
    data = load_dataset()

    measures_all_runs = []

    steps_per_epoch = None

    for run_id in range(n_runs):
        model = load_model(input_size=data.train.examples.shape[1], output_size=data.n_classes)
        os.makedirs("activations", exist_ok=True)
        file_name_all_activations = f'activations/activations_experiment_{_run._id}_run_{run_id}'
        file_all_activations = h5py.File(file_name_all_activations, "a")
        callbacks = make_callbacks(data=data, file_all_activations=file_all_activations)
        model.fit(x=data.train.examples, y=data.train.one_hot_labels,
                  verbose=2,
                  batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  validation_data=(data.test.examples, data.test.one_hot_labels),
                  callbacks=callbacks)

        print('fit successful')

        estimator = load_estimator(data=data)
        measures = estimator.compute_mi(file_all_activations=file_all_activations)
        measures['run'] = run_id
        measures_all_runs.append(measures)

        # Clear the current Session to free current layer and model definition.
        # This would otherwise be kept in memory. It is not needed as every run
        # redefines the model.
        K.clear_session()

    # Transform list of measurements into DataFrame with hierarchical index.
    measures_all_runs = pd.concat(measures_all_runs)
    measures_all_runs = measures_all_runs.fillna(0)

    # Save information measures
    mi_filename = "information_measures.csv"
    measures_all_runs.to_csv(mi_filename)
    _run.add_artifact(mi_filename, name="information_measures")

    # compute mean of information measures over all runs
    mi_mean_over_runs = measures_all_runs.groupby(['epoch', 'layer']).mean()

    measures_summary = {'measures_all_runs': measures_all_runs,
                        'mi_mean_over_runs': mi_mean_over_runs,
                        'activations_summary': file_all_activations}

    plotter_objects = make_plotters()
    generate_plots(plotter_objects, measures_summary)
