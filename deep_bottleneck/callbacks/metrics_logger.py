from tensorflow.python.keras.callbacks import Callback


class MetricsLogger(Callback):
    """Callback to log loss and accuracy to sacred database."""

    def __init__(self, file_dump, do_save_func):
        super().__init__()

        self._file_dump = file_dump
        self._do_save_func = do_save_func

    def on_epoch_end(self, epoch, logs=None):
        if self._do_save_func(epoch):
            self._file_dump.require_group(f'{epoch}/accuracy')
            self._file_dump.require_group(f'{epoch}/loss')

            self._file_dump[f'{epoch}/accuracy']['training'] = float(logs['acc'])
            self._file_dump[f'{epoch}/loss']['training'] = float(logs['acc'])

            try:
                self._file_dump[f'{epoch}/accuracy']['test'] = float(logs['val_acc'])
                self._file_dump[f'{epoch}/loss']['test'] = float(logs['val_loss'])

            except KeyError:
                print('Validation not enabled. Validation metrics cannot be logged')


class SacredMetricsLogger(Callback):

    def __init__(self, run):
        super().__init__()

        self._run = run

    def on_epoch_end(self, epoch, logs=None):

        self._run.log_scalar("training.loss", float(logs['loss']), step=epoch)
        self._run.log_scalar("training.accuracy", float(logs['acc']), step=epoch)

        try:
            self._run.log_scalar("test.loss", float(logs['val_loss']), step=epoch)
            self._run.log_scalar("test.accuracy", float(logs['val_acc']), step=epoch)

        except KeyError:
            print('Validation not enabled. Validation metrics cannot be logged')
