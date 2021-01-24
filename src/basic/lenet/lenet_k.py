# Instantiate Model
from tensorflow.keras import layers
import tensorflow.keras as keras

model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(5,5), activation='tanh', input_shape=(28,28,1)))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=16, kernel_size=(5,5), activation='tanh'))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=10, activation = 'softmax'))

loss = 'sparse_categorical_crossentropy'
optimizer = 'adam'

# Train model
if tf.test.is_gpu_available():
    print('Running with GPU')
    with tf.device('/device:GPU:0'):
        model.compile(
            optimizer='adam', 
            metrics=['accuracy'], 
            loss='sparse_categorical_crossentropy'
        )
        train_history = model.fit(
            train_x, train_y, 
            batch_size=batch_size, 
            epochs=50,
            validation_data=(val_x, val_y))
else: 
    print('Running with CPU')
    model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
    with tf.device('/device:CPU:0'):
        train_history = model.fit(
            train_x, 
            train_y, 
            batch_size=batch_size, 
            epochs=50,
            validation_data=(val_x, val_y)
        )
