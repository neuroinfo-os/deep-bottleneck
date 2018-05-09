import importlib

from sacred import Experiment
from sacred.observers import MongoObserver
from iclr_wrap_up.callbacks.loggingreporter import LoggingReporter
from sacred.observers import FileStorageObserver


ex = Experiment('sacred_keras_example')
#ex.observers.append(MongoObserver.create(url='mongodb://127.0.0.1:27017',
#                                         db_name='dneck_test'))


@ex.config
def hyperparameters():
    epochs = 10
    batch_size = 256
    architecture = [10, 7, 3]
    learning_rate = 0.0004
    full_mi = True
    infoplane_measure = 'upper'
    architecture_name= '-'.join(map(str, architecture))
    activation_fn = 'tanh'
    save_dir = 'rawdata/' + activation_fn + '_' + architecture_name
    infoplane_measure = 'upper'
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'
    estimator = 'compute_mi.compute_mi_ib_net'

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
    estimator.compute_mi()

