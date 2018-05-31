from tensorflow import keras
from tensorflow.python.keras import backend as K

import os

import tensorflow as tf
from tensorflow.python.summary import summary as tf_summary
from tensorflow.contrib.tensorboard.plugins import projector

import utils
import numpy as np


class ActivityProjector(keras.callbacks.Callback):

    def __init__(self, train, test, log_dir='./logs', embeddings_freq=10):
        super().__init__()

        self.sess = K.get_session()

        self.log_dir = log_dir
        self.embeddings_freq = embeddings_freq

        self.writer = tf_summary.FileWriter(self.log_dir)
        self.saver: tf.train.Saver

        self.embeddings_ckpt_path = os.path.join(self.log_dir, 'keras_embedding.ckpt')

        self.train = train
        self.test = test
        self.full = utils.construct_full_dataset(train, test)

        # Save metadata.
        np.savetxt(f'{log_dir}/metadata.tsv', self.test.y, fmt='%i')

    def set_model(self, model):
        self.model = model

        embeddings = []
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):  # Dense-like layers have a kernel attribute.
                layerfunc = K.function(self.model.inputs, [layer.output])
                layer_activity = layerfunc([self.test.X])[0]
                embeddings.append(tf.Variable(layer_activity, name=layer.name))

        self.saver = tf.train.Saver(embeddings)

        config = projector.ProjectorConfig()

        for tensor in embeddings:
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor.name
            embedding.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        if self.embeddings_freq and self.embeddings_ckpt_path:
            if (epoch % self.embeddings_freq) == 0:
                self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()