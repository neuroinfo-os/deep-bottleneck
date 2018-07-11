from tensorflow import keras
from tensorflow.python.keras.constraints import max_norm
import tensorflow as tf
import numpy as np

optimizer_map = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
}

activation_fn_map = {
    'tanh': tf.nn.tanh,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'softsign': tf.nn.softsign,
    'softplus': tf.nn.softplus,
    'leaky_relu': tf.nn.leaky_relu,
    'hard_sigmoid': 'hard_sigmoid',
    'selu': tf.nn.selu,
    'relu6': tf.nn.relu6,
    'elu': tf.nn.elu,
    'linear': 'linear'
}


def load(architecture, activation_fn, optimizer, learning_rate, input_size, output_size, max_norm_weights=False):
    input_layer = keras.layers.Input((input_size,))
    clayer = input_layer
    for n in architecture:
        clayer = keras.layers.Dense(n,
                                    activation=activation_fn_map[activation_fn],
                                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=1 / np.sqrt(float(n)),
                                                                                          seed=None),
                                    bias_initializer='zeros',
                                    kernel_constraint=(max_norm(max_value=float(max_norm_weights))
                                                       if max_norm_weights else None)
                                    )(clayer)
    output_layer = keras.layers.Dense(output_size, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizer_map[optimizer](learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
