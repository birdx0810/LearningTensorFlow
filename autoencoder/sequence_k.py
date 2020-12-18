import json
import os
import random

from tqdm import trange
import tensorflow as tf
import pandas as pd
import numpy as np

import utils

# Set random seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

# Get data
data = utils.get_anscombe_dataset()

# Set hyperparameters
learning_rate = 1e-3
epochs = 10
batch_size = 2

num_layers = 2
max_seq_len = 11
feature_dim = 2
hidden_dim = 10

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(hidden_dim, activation=tf.nn.tanh)
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(hidden_dim, activation=tf.nn.tanh)
                tf.keras.layers.Dense(feature_dim, activation=tf.nn.sigmoid)
            ]
        )

    def call(self, x):
        # Encoder
        h = self.encoder(x)
        # Decoder
        x_hat = self.decoder(h)
        return x_hat

model = AutoEncoder()
model.compile(
    optimizer=tf.keras.optimizers.Adam, 
    loss=tf.keras.losses.MeanSquaredError
)
model.fit(
    x=data, 
    y=data,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_data=(x_test, x_test)
)


