# Import required modules
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
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

# First define the model 
model = tf.keras.Sequential()

# Add the dense layer to the model
model.add(tf.keras.layers.Dense(
    units=num_labels, 
    activation="softmax", 
    input_shape=(784,), 
    weights=[np.ones([784, 10]), np.zeros(10)]
))

model.summary()

# Define loss and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
criterion = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, metrics=["accuracy"], loss=criterion)
