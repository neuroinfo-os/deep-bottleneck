import keras
import tensorflow as tf
import numpy as np

def load(architecture, activation_fn, learning_rate):
    input_layer = keras.layers.Input((trn.X.shape[1],))
    clayer = input_layer
    for n in architecture:
        clayer = keras.layers.Dense(n,
                                    activation=cfg['ACTIVATION'],
                                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                          stddev=1 / np.sqrt(float(n)),
                                                                                          seed=None),
                                    bias_initializer='zeros'
                                    )(clayer)
    output_layer = keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=cfg['SGD_LEARNINGRATE']))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])