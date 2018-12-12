from tensorflow import keras
from tensorflow.python.keras import backend as K

import os

import tensorflow as tf
from tensorflow.python.summary import summary as tf_summary
from tensorflow.contrib.tensorboard.plugins import projector

from deep_bottleneck import utils
import numpy as np


class ActivityProjector(keras.callbacks.Callback):
    """Read activity from layers of a Keras model and log is for TensorBoard

    This callback reads activity from the hidden layers of a Keras model
    and logs it as Model Checkpoint files.
    The network activity patterns can then be explored in TensorBoard
    with its Embeddings Projector
    """

    def __init__(self, test_set, log_dir='./logs', embeddings_freq=10):
        """
        Args:
            test_set: The test data
            log_dir: Path to directory used for logging
            embeddings_freq: Defines how often embedding variables will be saved to
                the log directory. If set to 1, this is done every epoch, if it is set to 10 every 10th epoch and so forth.
        """
        super().__init__()

        self.sess = K.get_session()

        self.log_dir = log_dir
        self.embeddings_freq = embeddings_freq

        self.writer = tf_summary.FileWriter(self.log_dir)
        self.saver: tf.train.Saver

        self.embeddings_ckpt_path = os.path.join(self.log_dir, 'keras_embedding.ckpt')

        self.test_set = test_set

        # Save metadata.
        np.savetxt(f'{log_dir}/metadata.tsv', self.test_set.labels, fmt='%i')

    def set_model(self, model):
        """Prepare for logging the activities of the layers and set up the TensorBoard projector
        Args:
            model: The Keras model

        Returns:
            None
        """
        self.model = model

        embeddings = []
        for layer in self.model.layers:
            if utils.is_dense_like(layer):
                layerfunc = K.function(self.model.inputs, [layer.output])
                layer_activity = layerfunc([self.test_set.examples])[0]
                embeddings.append(tf.Variable(layer_activity, name=layer.name))

        self.saver = tf.train.Saver(embeddings)

        config = projector.ProjectorConfig()

        for tensor in embeddings:
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name
            embedding.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        """Write layer activations to file
        Args:
            epoch: Number of the current epoch
            logs: Quantities such as acc, loss which are passed by Sequential.fit()

        Returns:
            None
        """
        if self.embeddings_freq and self.embeddings_ckpt_path:
            if (epoch % self.embeddings_freq) == 0:
                self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)

    def on_train_end(self, logs=None):
        """Close files
        Args:
            logs: Quantities such as acc, loss which are passed by Sequential.fit()

        Returns:
            None
        """
        self.writer.close()
