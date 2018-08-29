from tensorflow import keras
from tensorflow.python.keras import backend as K
import numpy as np

from deep_bottleneck import utils


class LoggingReporter(keras.callbacks.Callback):
    def __init__(self, dataset, calculate_mi_for, batch_size, activation_fn, file_all_activations, do_save_func=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.calculate_mi_for = calculate_mi_for
        self.batch_size = batch_size
        self.activation_fn = activation_fn
        self.file_all_activations = file_all_activations

        # do_save_func(epoch) should return True if we should save on that epoch.
        self.do_save_func = do_save_func

    def on_train_begin(self, logs=None):

        # Indexes of the layers which we keep track of. Basically, this will be any layer 
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layer_indexes = []

        # Functions return activity of each layer
        self.layerfuncs = []

        # Functions return weights of each layer
        self.layerweights = []
        for layer_idx, layer in enumerate(self.model.layers):
            if utils.is_dense_like(layer):
                self.layer_indexes.append(layer_idx)
                self.layerfuncs.append(K.function(self.model.inputs, [layer.output]))
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
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients

            self._log_gradients = True
            self._batch_weightnorm = []

            self._batch_gradients = [[] for _ in self.model.layers[1:]]

            # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
            ixs = list(range(len(self.dataset.train.examples)))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_batch_begin(self, batch, logs=None):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return

        # Sample a batch
        batchsize = self.batch_size
        cur_ixs = self._batch_todo_ixs[:batchsize]
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]

        # Get gradients for this batch
        inputs = [self.dataset.train.examples[cur_ixs, :],  # Inputs
                  [1] * len(cur_ixs),  # Uniform sample weights
                  self.dataset.train.one_hot_labels[cur_ixs, :],  # Outputs
                  1  # Training phase
                  ]
        for lndx, g in enumerate(self.get_gradients(inputs)):
            # g is gradients for weights of lndx's layer
            oneDgrad = np.reshape(g, -1, 1)  # Flatten to one dimensional vector
            self._batch_gradients[lndx].append(oneDgrad)

    def on_epoch_end(self, epoch, logs=None):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return

        # Get overall performance
        loss = {}
        for data, data_split, istrain in ((self.dataset.train, 'trn', 1), (self.dataset.test, 'tst', 0)):
            loss[data_split] = self.get_loss([data.examples,
                                             [1] * len(data.examples),
                                             data.one_hot_labels, istrain])[0].flat[0]

        self.file_all_activations.create_group(str(epoch))
        self.file_all_activations[str(epoch)].create_dataset('weights_norm', (len(self.layer_indexes),))
        self.file_all_activations[str(epoch)].create_dataset('gradmean', (len(self.layer_indexes),))
        self.file_all_activations[str(epoch)].create_dataset('gradstd', (len(self.layer_indexes),))
        self.file_all_activations[str(epoch)].create_group('activations')

        for i, layerix in enumerate(self.layer_indexes):
            layer = self.model.layers[layerix]

            self.file_all_activations[f'{epoch}/weights_norm'][i] = np.linalg.norm(K.get_value(layer.kernel))

            stackedgrads = np.stack(self._batch_gradients[i], axis=1)
            self.file_all_activations[f'{epoch}/gradmean'][i] = np.linalg.norm(stackedgrads.mean(axis=1))
            self.file_all_activations[f'{epoch}/gradstd'][i] = np.linalg.norm(stackedgrads.std(axis=1))

            # TODO Same "if" clause is in the estimatior, remove code duplication
            if self.calculate_mi_for == "test":
                self.file_all_activations[f'{epoch}/activations/'].create_dataset(
                    str(i),
                    data=self.layerfuncs[i]([self.dataset.test.examples])[0]
                )
            elif self.calculate_mi_for == "training":
                self.file_all_activations[f'{epoch}/activations/'].create_dataset(
                    str(i),
                    data=self.layerfuncs[i]([self.dataset.train.examples])[0]
                )
