import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Import the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
train_images = x_train.reshape(60000, 784).astype('float32') / 255.0
test_images = x_test.reshape(10000, 784).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(y_train, 10)
test_labels = tf.keras.utils.to_categorical(y_test, 10)

# Define the neural network architecture
input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# Define the neural network model
hidden_nodes = 512
input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))
hidden_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

input_layer = tf.matmul(input_images, input_weights) + input_biases
hidden_layer = tf.nn.relu(input_layer)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

# Define the loss function and optimizer
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))

# Define methods to measure accuracy
correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a session
sess = tf.compat.v1.Session()

# Define different batch sizes
batch_sizes = [64, 128, 256]  # Example: List of different batch sizes

for batch_size in batch_sizes:
    # Calculate the number of batches
    num_batches = len(train_images) // batch_size

    # Define the optimizer with the current batch size
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss_function)

    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train the neural network
    for epoch in range(20):
        avg_loss = 0.0
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            batch_X = train_images[start_idx:end_idx]
            batch_Y = train_labels[start_idx:end_idx]
            _, loss = sess.run([optimizer, loss_function], feed_dict={input_images: batch_X, target_labels: batch_Y})
            avg_loss += loss / num_batches

        print("Epoch:", epoch + 1, " Loss:", avg_loss)

    # Calculate and print the accuracy on the test data
    test_acc = sess.run(accuracy, feed_dict={input_images: test_images, target_labels: test_labels})
    print("Batch Size:", batch_size, " Accuracy:", test_acc)
    print()

