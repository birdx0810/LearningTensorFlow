# Import required modules
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow.compat.v1 as tf
tf.set_random_seed(42)
tf.disable_v2_behavior()

from tensorflow.keras.datasets import mnist
from tqdm import trange

# Set random seed
np.random.seed(42)
tf.set_random_seed(42)

# Set Model Parameters
learning_rate = 1e-5 
epochs = 10
batch_size = 32
num_labels = 10

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1]*train_x.shape[2])
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1]*test_x.shape[2])

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(num_labels))

train_onehot = label_binarizer.transform(train_y)
test_onehot = label_binarizer.transform(test_y)

# tf Graph Input 
x = tf.placeholder("float", [None, 784])  # mnist data image of shape 28*28 = 784 
y = tf.placeholder("float", [None, 10])   # 0-9 digits recognition => 10 classes

# Set model weight and bias 
W = tf.Variable(tf.ones([784, 10]))      # 784 -> 10 
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

pred = tf.nn.softmax(logits)
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y,1), predictions=tf.argmax(pred,1))

# Optimize model using gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initializing the variables 
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

# Launch the graph 
with tf.Session() as sess:
    sess.run(init_global)
    sess.run(init_local)
  
    # Training cycle
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.
        iterations = int(len(train_x)/batch_size)
        # Loop over all batches
        for i in trange(iterations):
            # Get mini batch
            batch_xs, batch_ys = get_batch(train_x, train_onehot, i, batch_size)
      
            # Fit training using batch data 
            _, batch_loss = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})

            total_loss += batch_loss/iterations

        # Display logs per epoch step
        total_loss = total_loss
        print(f"\nEpoch:{epoch+1}\ttotal loss={total_loss}")
    
    _, accuracy = sess.run([acc, acc_op], feed_dict={x: test_x, y: test_onehot})
    print(f"model acc: {accuracy}")
    print(f"Time taken: {time.time() - start} sec")

