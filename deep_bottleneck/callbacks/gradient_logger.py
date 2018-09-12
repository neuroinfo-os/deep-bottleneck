from tensorflow import keras
from tensorflow.python.keras import backend as K
import numpy as np

from deep_bottleneck import utils


class GradientLogger(keras.callbacks.Callback):
    def __init__(self, data, batch_size, file_dump, do_save_func=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.batch_size = batch_size
        self.file_dump = file_dump
        self._log_gradients = False

        # do_save_func(epoch) should return True if we should save on that epoch.
        self.do_save_func = do_save_func

    def on_train_begin(self, logs=None):

        # Indexes of the layers which we keep track of. Basically, this will be any layer
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layer_indexes = []

        # Functions return weights of each layer
        self.layerweights = []
        for layer_idx, layer in enumerate(self.model.layers):
            if utils.is_dense_like(layer):
                self.layer_indexes.append(layer_idx)
                self.layerweights.append(layer.kernel)

        input_tensors = [self.model.inputs[0],
                         self.model.sample_weights[0],
                         self.model.targets[0],
                         K.learning_phase()]
        # Get gradients of all the relevant layers at once
        grads = self.model.optimizer.get_gradients(self.model.total_loss, self.layerweights)
        self.get_gradients = K.function(inputs=input_tensors,
                                        outputs=grads)

        # Get cross-entropy loss
        self.get_loss = K.function(inputs=input_tensors, outputs=[self.model.total_loss])

    def on_epoch_begin(self, epoch, logs=None):
        if self.do_save_func(epoch):
            # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients

            self._log_gradients = True
            self._batch_weightnorm = []

            self._batch_gradients = [[] for _ in self.model.layers[1:]]

            # Indexes of all the training data samples. These are shuffled and read-in in chunks of self.batch_size.
            example_indexes = list(range(len(self.data.examples)))
            np.random.shuffle(example_indexes)
            self._batch_todo_indexes = example_indexes

        else:
            # Don't log this epoch
            self._log_gradients = False

    def on_batch_begin(self, batch, logs=None):
        if self._log_gradients:
            # Sample a batch
            current_indexes = self._batch_todo_indexes[:self.batch_size]
            # Advance the indexing, so next on_batch_begin samples a different batch
            self._batch_todo_indexes = self._batch_todo_indexes[self.batch_size:]

            # Get gradients for this batch
            inputs = [self.data.examples[current_indexes, :],  # Inputs
                      [1] * len(current_indexes),  # Uniform sample weights
                      self.data.one_hot_labels[current_indexes, :],  # Outputs
                      1  # Training phase
                      ]
            for layer_index, gradients in enumerate(self.get_gradients(inputs)):
                gradients_flattened = gradients.flatten()
                self._batch_gradients[layer_index].append(gradients_flattened)

    def on_epoch_end(self, epoch, logs=None):
        if self.do_save_func(epoch):

            self.file_dump.require_group(str(epoch))
            self.file_dump[str(epoch)].create_dataset('weights_norm', (len(self.layer_indexes),))
            self.file_dump[str(epoch)].create_dataset('gradmean', (len(self.layer_indexes),))
            self.file_dump[str(epoch)].create_dataset('gradstd', (len(self.layer_indexes),))

            for i, layer_index in enumerate(self.layer_indexes):
                layer = self.model.layers[layer_index]

                self.file_dump[f'{epoch}/weights_norm'][i] = np.linalg.norm(K.get_value(layer.kernel))

                stacked_gradients = np.stack(self._batch_gradients[i], axis=1)
                self.file_dump[f'{epoch}/gradmean'][i] = np.linalg.norm(stacked_gradients.mean(axis=1))
                self.file_dump[f'{epoch}/gradstd'][i] = np.linalg.norm(stacked_gradients.std(axis=1))
