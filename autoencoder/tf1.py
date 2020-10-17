import json
import os
import random

from tqdm import trange
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
input_dim = 2
hidden_dim = 10

# Prepare Graph
def model(data):
    enc_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    dec_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=input_dim, activation=tf.nn.tanh)

    tf.reset_default_graph()

    # Input place holders
    X = tf.placeholder(tf.float32, [None, max_seq_len, input_dim])
    S = tf.placeholder(tf.float32, [None])

    def encoder (X, S):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=S)
            H = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid)
        return H
        
    def decoder (H, S):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            r_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=input_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=S)
            X_tilde = tf.layers.dense(r_outputs, input_dim, activation=None)
        return X_tilde

    H = encoder(X, S)
    X_tilde = decoder(H, S)

    loss = tf.losses.mean_squared_error(X, X_tilde)
    scaled = 10 * tf.sqrt(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(scaled)

    # Initializing the variables 
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    # Launch the graph 
    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
    
        # Training cycle
        for epoch in range(epochs):
            total_loss = 0.
            iterations = int(len(data)/batch_size)

            for i in trange(iterations):
                # Get mini batch
                batch_xs, batch_ts = utils.get_batch(data, i, batch_size)

                # Fit training using batch data 
                _, batch_loss = sess.run([optimizer, scaled], feed_dict={X: batch_xs, S: batch_ts})

                total_loss += batch_loss/iterations
      
            print(f"\nEpoch: {epoch}\tLoss: {total_loss}")

model(data)
