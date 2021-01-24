import tensorflow.v1.compat as tf

# tf Graph Input 
x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28 = 784 
y = tf.placeholder("float", [None, 10])   # 0-9 digits recognition => 10 classes

# variables for trainable weights
conv1_weights = tf.Variable(
    tf.truncated_normal(
        [5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
        stddev=0.1,
        seed=SEED, 
        dtype=data_type()
    )
)
conv1_biases = tf.Variable(
    tf.zeros(
        [32], 
        dtype=data_type()
    )
)
conv2_weights = tf.Variable(
    tf.truncated_normal(
    [5, 5, 32, 64], 
    stddev=0.1,
    seed=SEED, 
    dtype=data_type()
    )
)
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
        stddev=0.1,
        seed=SEED,
        dtype=data_type()
    )
)
fc1_biases = tf.Variable(
    tf.constant(
        0.1, 
        shape=[512], 
        dtype=data_type()
    )
)
fc2_weights = tf.Variable(
    tf.truncated_normal(
        [512, NUM_LABELS],
        stddev=0.1,
        seed=SEED,
        dtype=data_type()
    )
)
fc2_biases = tf.Variable(
    tf.constant(
      0.1, 
      shape=[NUM_LABELS], 
      dtype=data_type()
    )
)

def LeNet(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(
        data,
        conv1_weights,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )
    conv = tf.nn.conv2d(
        pool,
        conv2_weights,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(
        relu,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

# Initializing the variables 
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# Launch the graph 
with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    sess.run(init_global)
    sess.run(init_local)

    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                    train_labels_node: batch_labels}
        # Run the optimizer to update weights.
        sess.run(optimizer, feed_dict=feed_dict)

