import keras
import tensorflow as tf
import numpy as np

def load(architecture, activation_fn, learning_rate, input_size, output_size):
    input_layer = keras.layers.Input((input_size,))
    clayer = input_layer
    for n in architecture:
        clayer = keras.layers.Dense(n,
                                    activation=activation_fn,
                                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=1 / np.sqrt(float(n)),
                                                                                          seed=None),
                                    bias_initializer='zeros'
                                    )(clayer)
    output_layer = keras.layers.Dense(output_size, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=learning_rate))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model