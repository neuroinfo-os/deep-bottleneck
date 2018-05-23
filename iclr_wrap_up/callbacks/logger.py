from tensorflow.python.keras.callbacks import Callback

class Logger(Callback):

    def __init__(self, run):
        super().__init__()

        self._run = run

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        self._run.log_scalar("training.loss",  float(logs['loss']))