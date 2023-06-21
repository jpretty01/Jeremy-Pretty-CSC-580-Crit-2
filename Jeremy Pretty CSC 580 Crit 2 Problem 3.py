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

# Define the neural network architecture
input_images = tf.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

# Define a function to create the neural network model
def create_model(hidden_nodes):
    input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
    input_biases = tf.Variable(tf.zeros([hidden_nodes]))
    hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
    hidden_biases = tf.Variable(tf.zeros([10]))

    input_layer = tf.matmul(input_images, input_weights) + input_biases
    hidden_layer = tf.nn.relu(input_layer)
    digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

    return digit_weights

# Train and evaluate the neural network with different numbers of hidden neurons
hidden_neurons_list = [128, 256, 512, 1024]  # Example: List of different numbers of hidden neurons

for hidden_neurons in hidden_neurons_list:
    # Build the model with the current number of hidden neurons
    digit_weights = create_model(hidden_neurons)

    # Define the loss function and optimizer
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=target_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

    # Define methods to measure accuracy
    correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a session and initialize variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train the neural network
    for epoch in range(20):
        _, loss = sess.run([optimizer, loss_function], feed_dict={input_images: train_images, target_labels: train_labels})
        print("Epoch:", epoch + 1, " Loss:", loss)

    # Calculate and print the accuracy on the test data
    test_accuracy = sess.run(accuracy, feed_dict={input_images: test_images, target_labels: test_labels})
    print("Hidden Neurons:", hidden_neurons, " Accuracy:", test_accuracy)
    print()
