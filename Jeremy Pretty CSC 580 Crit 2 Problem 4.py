# Jeremy Pretty
# CSC 580 Crit 2 Problem 4
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
input_images = tf.keras.Input(shape=input_shape)
target_labels = tf.keras.Input(shape=(10,))

# Define the neural network model
hidden_nodes = 512
hidden_layer = tf.keras.layers.Dense(hidden_nodes, activation='relu')(input_images)
digit_weights = tf.keras.layers.Dense(10)(hidden_layer)

# Define the loss function
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Define methods to measure accuracy
accuracy = tf.keras.metrics.CategoricalAccuracy()

# Train and evaluate the neural network with different learning rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]  # Example: List of different learning rates

for learning_rate in learning_rates:
    # Define the optimizer with the current learning rate
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize variables
    tf.keras.backend.clear_session()

    # Create the model
    model = tf.keras.Model(inputs=input_images, outputs=digit_weights)

    # Train the neural network
    for epoch in range(20):
        with tf.GradientTape() as tape:
            logits = model(train_images, training=True)
            loss_value = loss_function(train_labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update the accuracy metric
        accuracy.update_state(train_labels, logits)

    # Evaluate the model on the test data
    test_logits = model(test_images, training=False)
    test_accuracy = accuracy(test_labels, test_logits)

    print("Learning Rate:", learning_rate, " Accuracy:", test_accuracy.numpy())
    print()
