# Jeremy Pretty
# CSC 580 Crit 2 Problem 3
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

# Define a function to create the neural network model
def create_model(hidden_nodes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_nodes, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10)
    ])
    return model

# Train and evaluate the neural network with different numbers of hidden neurons
hidden_neurons_list = [128, 256, 512, 1024]  # Example: List of different numbers of hidden neurons

for hidden_neurons in hidden_neurons_list:
    # Build the model with the current number of hidden neurons
    model = create_model(hidden_neurons)

    # Define the loss function and optimizer
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.5)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Train the neural network
    model.fit(train_images, train_labels, epochs=20, batch_size=32, verbose=0)

    # Evaluate the model on the test data
    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print("Hidden Neurons:", hidden_neurons, " Accuracy:", test_accuracy)
    print()
