from tensorflow import keras
import tensorflow as tf
import numpy as np

optimizer_map = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
}

def load(architecture, activation_fn, optimizer, learning_rate, input_size, output_size):
    input_layer = keras.layers.Input((input_size,))
    clayer = input_layer
    for n in architecture:
        clayer = keras.layers.Dense(n,
                                    activation=activation_fn,
                                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=1 / np.sqrt(float(n)),
                                                                                          seed=None),
                                    bias_initializer='zeros',
                                    )(clayer)
    output_layer = keras.layers.Dense(output_size, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizer_map[optimizer](learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
