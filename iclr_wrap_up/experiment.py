import importlib

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorflow.python.keras.callbacks import Callback


ex = Experiment('sacred_keras_example')
ex.observers.append(MongoObserver.create(url='mongodb://127.0.0.1:27017',
                                         db_name='dneck_test'))


class Logger(Callback):

    def __init__(self, run):
        super().__init__()

        self._run = run

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        self._run.log_scalar("training.loss",  float(logs['loss']))


@ex.config
def hyperparameters():
    epochs = 10
    batch_size = 32
    architecture = [10, 7, 3]
    model = 'models.feedforward'
    dataset = 'datasets.harmonics'

@ex.capture
def load_dataset(dataset):
    module = importlib.import_module(dataset)
    return module.load()

@ex.capture
def load_model(model, architecture):
    module = importlib.import_module(model)
    return module.load(architecture)

@ex.capture
def make_callbacks(_run):
    return [Logger(_run)]

@ex.automain
def conduct(epochs, batch_size):
    dataset = load_dataset()
    model = load_model()
    callbacks = make_callbacks()
    model.fit(dataset, dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks)





