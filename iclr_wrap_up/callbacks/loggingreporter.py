from tensorflow import keras
from tensorflow.python.keras import backend as K
import numpy as np

from collections import OrderedDict
from iclr_wrap_up import utils


class LoggingReporter(keras.callbacks.Callback):
    def __init__(self, trn, tst, calculate_mi_for, batch_size, activation_fn, file_all_activations, do_save_func=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trn = trn  # Train data
        self.tst = tst  # Test data
        self.calculate_mi_for = calculate_mi_for
        self.batch_size = batch_size
        self.activation_fn = activation_fn
        self.file_all_activations = file_all_activations

        if self.calculate_mi_for == "full_dataset":
            self.full = utils.construct_full_dataset(trn, tst)

        # do_save_func(epoch) should return True if we should save on that epoch
        self.do_save_func = do_save_func

    def on_train_begin(self, logs={}):

        # Indexes of the layers which we keep track of. Basically, this will be any layer 
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layerixs = []

        # Functions return activity of each layer
        self.layerfuncs = []

        # Functions return weights of each layer
        self.layerweights = []
        for lndx, l in enumerate(self.model.layers):
            if utils.is_dense_like(l):
                self.layerixs.append(lndx)
                self.layerfuncs.append(K.function(self.model.inputs, [l.output]))
                self.layerweights.append(l.kernel)

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

    def on_epoch_begin(self, epoch, logs={}):
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
            ixs = list(range(len(self.trn.X)))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_batch_begin(self, batch, logs={}):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return

        # Sample a batch
        batchsize = self.batch_size
        cur_ixs = self._batch_todo_ixs[:batchsize]
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]

        # Get gradients for this batch
        inputs = [self.trn.X[cur_ixs, :],  # Inputs
                  [1] * len(cur_ixs),  # Uniform sample weights
                  self.trn.Y[cur_ixs, :],  # Outputs
                  1  # Training phase
                  ]
        for lndx, g in enumerate(self.get_gradients(inputs)):
            # g is gradients for weights of lndx's layer
            oneDgrad = np.reshape(g, -1, 1)  # Flatten to one dimensional vector
            self._batch_gradients[lndx].append(oneDgrad)

    def on_epoch_end(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return

        # Get overall performance
        loss = {}
        for cdata, cdataname, istrain in ((self.trn, 'trn', 1), (self.tst, 'tst', 0)):
            loss[cdataname] = self.get_loss([cdata.X, [1] * len(cdata.X), cdata.Y, istrain])[0].flat[0]

        self.file_all_activations.create_group(str(epoch))
        self.file_all_activations[str(epoch)].create_dataset('weights_norm', (len(self.layerixs),))
        self.file_all_activations[str(epoch)].create_dataset('gradmean', (len(self.layerixs),))
        self.file_all_activations[str(epoch)].create_dataset('gradstd', (len(self.layerixs),))
        self.file_all_activations[str(epoch)].create_group('activations')

        for lndx, layerix in enumerate(self.layerixs):
            clayer = self.model.layers[layerix]

            self.file_all_activations[f'{epoch}/weights_norm'][lndx] = np.linalg.norm(K.get_value(clayer.kernel))

            stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            self.file_all_activations[f'{epoch}/gradmean'][lndx] = np.linalg.norm(stackedgrads.mean(axis=1))
            self.file_all_activations[f'{epoch}/gradstd'][lndx] = np.linalg.norm(stackedgrads.std(axis=1))

            # TODO Same "if" clause is in the estimatior, remove code duplication
            if self.calculate_mi_for == "full_dataset":
                self.file_all_activations[f'{epoch}/activations/'].create_dataset(str(lndx), data=
                        self.layerfuncs[lndx]([self.full.X])[0])
            elif self.calculate_mi_for == "test":
                self.file_all_activations[f'{epoch}/activations/'].create_dataset(str(lndx), data=
                        self.layerfuncs[lndx]([self.tst.X])[0])
            elif self.calculate_mi_for == "training":
                self.file_all_activations[f'{epoch}/activations/'].create_dataset(str(lndx), data=
                        self.layerfuncs[lndx]([self.trn.X])[0])