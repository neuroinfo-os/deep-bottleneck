from tensorflow.python.keras.callbacks import Callback


class MetricsLogger(Callback):
    """Callback to log loss and accuracy to sacred database."""
    def __init__(self, run):
        super().__init__()

        self._run = run

    def on_epoch_end(self, epoch, logs=None):
        self._run.log_scalar("training.loss", float(logs['loss']), step=epoch)
        self._run.log_scalar("training.accuracy", float(logs['acc']), step=epoch)

        try:
            self._run.log_scalar("validation.loss", float(logs['val_loss']), step=epoch)
            self._run.log_scalar("validation.accuracy", float(logs['val_acc']), step=epoch)
        except KeyError:
            print('Validation not enabled. Validation metrics cannot be logged')
