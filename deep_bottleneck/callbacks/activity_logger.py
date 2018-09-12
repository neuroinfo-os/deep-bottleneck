from tensorflow import keras
from tensorflow.python.keras import backend as K

from deep_bottleneck import utils


class ActivityLogger(keras.callbacks.Callback):
    def __init__(self, examples, file_dump, do_save_func=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.file_dump = file_dump
        # do_save_func(epoch) should return True if we should save on that epoch.
        self.do_save_func = do_save_func
        # Functions return activity of each layer.
        self.layerfuncs = []

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if utils.is_dense_like(layer):
                self.layerfuncs.append(K.function(self.model.inputs, [layer.output]))

    def on_epoch_end(self, epoch, logs=None):
        if self.do_save_func is not None and self.do_save_func(epoch):
            self._log_activity_for_epoch(epoch)

    def _log_activity_for_epoch(self, epoch):
        self.file_dump.require_group(str(epoch))
        self.file_dump[str(epoch)].create_group('activations')
        for i, current_layer_func in enumerate(self.layerfuncs):
            self.file_dump[str(epoch)]['activations'].create_dataset(
                name=str(i),
                data=current_layer_func([self.examples])[0]
            )
