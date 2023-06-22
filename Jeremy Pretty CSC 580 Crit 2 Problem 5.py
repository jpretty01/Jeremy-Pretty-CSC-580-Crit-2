# Jeremy Pretty
# CSC 580 Crit 2 Problem 5
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
train_images = x_train.reshape(60000, 784).astype('float32') / 255.0
test_images = x_test.reshape(10000, 784).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(y_train, 10)
test_labels = tf.keras.utils.to_categorical(y_test, 10)

# Define the neural network architecture
input_shape = (784,)
target_shape = (10,)

# Define the neural network model
hidden_nodes = 512
hidden_nodes_2 = 256

input_layer = tf.keras.layers.Input(shape=input_shape)
hidden_layer = tf.keras.layers.Dense(hidden_nodes, activation='relu')(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(hidden_nodes_2, activation='relu')(hidden_layer)
output_layer = tf.keras.layers.Dense(target_shape[0])(hidden_layer_2)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Define the loss function and optimizer
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=20)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Accuracy:", test_accuracy)
