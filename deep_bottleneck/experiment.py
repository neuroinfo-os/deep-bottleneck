import importlib
import pandas as pd
import numpy as np
import h5py
import os

from random import randint

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.callbacks import TensorBoard
from deep_bottleneck.callbacks.activity_logger import ActivityLogger

from deep_bottleneck.callbacks.gradient_logger import GradientLogger

from deep_bottleneck.callbacks.metrics_logger import MetricsLogger, SacredMetricsLogger
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
    initial_bias = 0.0
    if dataset == 'dataset.mnist':
        plotters = [
            # ('plotter.informationplane', []),
            #         ('plotter.snr', []),
            #         ('plotter.informationplane_movie', []),
            ('plotter.activations', [])
        ]
    else:
        plotters = [
            ('plotter.informationplane', []),
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
def load_model(model, architecture, activation_fn, optimizer,
               learning_rate, input_size, output_size, max_norm_weights, initial_bias):
    module = importlib.import_module(model)
    return module.load(architecture, activation_fn, optimizer,
               learning_rate, input_size, output_size, max_norm_weights, initial_bias)


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
def generate_plots(plotter_objects, measures_summary, suffix):
    for plotter in plotter_objects:
        plotter.generate(measures_summary, suffix)


@ex.capture
def make_callbacks(callbacks, data, batch_size, _run,
                   file_dump_train,
                   file_dump_test):
    callback_objects = []

    callback_objects.append(ActivityLogger(data.train.examples,
                                           do_save_func=do_report,
                                           file_dump=file_dump_train))

    callback_objects.append(ActivityLogger(data.test.examples,
                                           do_save_func=do_report,
                                           file_dump=file_dump_test))

    callback_objects.append(GradientLogger(data.train,
                                           batch_size=batch_size,
                                           do_save_func=do_report,
                                           file_dump=file_dump_train))

    callback_objects.append(GradientLogger(data.test,
                                           batch_size=batch_size,
                                           do_save_func=do_report,
                                           file_dump=file_dump_test))

    callback_objects.append(MetricsLogger(file_dump_train, do_report))
    callback_objects.append(MetricsLogger(file_dump_test, do_report))

    callback_objects.append(SacredMetricsLogger(_run))

    callback_objects.append(ActivityProjector(data.test,
                                              log_dir=f'./logs/{_run._id}',
                                              embeddings_freq=10))

    for callback in callbacks:
        callback_object = importlib.import_module(callback[0]).load(*callback[1])
        callback_objects.append(callback_object)

    return callback_objects


@ex.capture
def load_estimator(estimator, discretization_range, architecture, n_classes):
    module = importlib.import_module(estimator)
    return module.load(discretization_range, architecture, n_classes)

def generator(examples, labels, batch_size):
# Create empty arrays to contain batch of features and labels#

    batch_features = []
    batch_labels = []

    while True:
        for i in range(batch_size):
            # choose random index in features
            index= randint(0,len(examples)-1)
            batch_features.append(examples[index])
            batch_labels.append(labels[index])
        yield np.asarray(batch_features), np.asarray(batch_labels)   


@ex.automain
def conduct(epochs, batch_size, n_runs, _run, steps_per_epoch=None):
    data = load_dataset()

    plotter_objects = make_plotters()

    measures_all_runs_train = []
    measures_all_runs_test = []

    for run_id in range(n_runs):
        estimator = load_estimator(n_classes=data.n_classes)
        model = load_model(input_size=data.train.examples.shape[1], output_size=data.n_classes)
        os.makedirs("dumps", exist_ok=True)
        file_name_dump_train = f'dumps/experiment_{_run._id}_run_{run_id}_train.h5'
        file_name_dump_test = f'dumps/experiment_{_run._id}_run_{run_id}_test.h5'

        file_dump_train = h5py.File(file_name_dump_train, "a")
        file_dump_test = h5py.File(file_name_dump_test, "a")

        callbacks = make_callbacks(data=data,
                                   file_dump_train=file_dump_train,
                                   file_dump_test=file_dump_test)
        
        if steps_per_epoch == None:
            model.fit(x=data.train.examples, y=data.train.one_hot_labels,
                      verbose=2,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(data.test.examples, data.test.one_hot_labels),
                      callbacks=callbacks)
        else:    
            model.fit_generator(generator(data.train.examples, data.train.one_hot_labels, batch_size),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                validation_data=generator(data.test.examples, data.test.one_hot_labels, batch_size),
                                validation_steps = steps_per_epoch,
                                callbacks=callbacks)
        
        print('fit successful')

        measures_train = estimator.compute_mi(data.train,
                                              file_dump=file_dump_train)
        measures_train['run'] = run_id
        measures_all_runs_train.append(measures_train)

        measures_test = estimator.compute_mi(data.test,
                                             file_dump=file_dump_test)
        measures_test['run'] = run_id
        measures_all_runs_test.append(measures_test)

        # Clear the current Session to free current layer and model definition.
        # This would otherwise be kept in memory. It is not needed as every run
        # redefines the model.
        K.clear_session()

    # Transform list of measurements into DataFrame with hierarchical index.
    measures_all_runs_train = pd.concat(measures_all_runs_train)
    measures_all_runs_train = measures_all_runs_train.fillna(0)

    measures_all_runs_test = pd.concat(measures_all_runs_test)
    measures_all_runs_test = measures_all_runs_test.fillna(0)

    # Save information measures
    mi_filename = "information_measures_train.csv"
    measures_all_runs_train.to_csv(mi_filename)
    _run.add_artifact(mi_filename, name="information_measures_train")

    mi_filename = "information_measures_test.csv"
    measures_all_runs_test.to_csv(mi_filename)
    _run.add_artifact(mi_filename, name="information_measures_test")

    # compute mean of information measures over all runs
    mi_mean_over_runs_train = measures_all_runs_train.groupby(['epoch', 'layer']).mean()
    mi_mean_over_runs_test = measures_all_runs_test.groupby(['epoch', 'layer']).mean()

    measures_summary_train = {'measures_all_runs': measures_all_runs_train,
                              'mi_mean_over_runs': mi_mean_over_runs_train,
                              'activations_summary': file_dump_train}

    measures_summary_test = {'measures_all_runs': measures_all_runs_test,
                             'mi_mean_over_runs': mi_mean_over_runs_test,
                             'activations_summary': file_dump_test}

    generate_plots(plotter_objects, measures_summary_train, suffix='train')
    generate_plots(plotter_objects, measures_summary_test, suffix='test')
