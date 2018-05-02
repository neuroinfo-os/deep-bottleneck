import importlib

from sacred import Experiment
from sacred.observers import MongoObserver
from .loggingreporter import LoggingReporter


ex = Experiment('sacred_keras_example')
ex.observers.append(MongoObserver.create(url='mongodb://127.0.0.1:27017',
                                         db_name='dneck_test'))


@ex.config
def hyperparameters():
    epochs = 10000
    batch_size = 256
    architecture = [10, 7, 3]
    learningrate = 0.0004
    activation_fn = 'relu'
    full_mi = True
    architecture_name= '-'.join(map(str, architecture))
    save_dir = 'rawdata/' + activation_fn + '_' + architecture_name
    infoplane_measure = 'upper'
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'
    estimator = 'compute_mi.compute_mi_bi_net'

@ex.capture
def load_dataset(dataset):
    module = importlib.import_module(dataset)
    return module.load()

@ex.capture
def load_model(model, architecture):
    module = importlib.import_module(model)
    return module.load(architecture)


def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 5 == 0)
    elif epoch < 2000:    # Then every 10th
        return (epoch % 20 == 0)
    else:                # Then every 100th
        return (epoch % 100 == 0)


@ex.capture
def make_callbacks(_run, training, test, full_mi, save_dir, batch_size, activation_fn):
    callbacks = [LoggingReporter(trn=training, tst=test, full_mi=full_mi, save_dir=save_dir,
                                 batch_size=batch_size, activation_fn= activation_fn, do_save_func=do_report)]
    return callbacks

@ex.capture
def load_estimator(estimator, full_mi):
    module = importlib.import_module(estimator)
    cls = module.load()
    return cls(full_mi)

@ex.automain
def conduct(epochs, batch_size):
    training, test = load_dataset()
    model = load_model()
    callbacks = make_callbacks(training, test)
    model.fit(x=training.X, y=training.Y,
                  verbose=2,
                  batch_size=batch_size,
                  epochs=epochs,
                  # validation_data=(tst.X, tst.Y),
                  callbacks=callbacks)
    estimator = load_estimator()
    estimator.compute_mi()







