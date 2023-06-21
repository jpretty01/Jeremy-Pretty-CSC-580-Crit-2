# Jeremy Pretty 
# CSC 580 Crit 2 Problem 2
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

hidden_nodes = 512
input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))
hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

input_layer = tf.matmul(input_images, input_weights) + input_biases
hidden_layer = tf.nn.relu(input_layer)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

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

# Find some misclassified images
predictions = tf.argmax(digit_weights, 1)
true_labels = tf.argmax(target_labels, 1)
misclassified_indices = []

for i in range(len(test_images)):
    predicted_label = sess.run(predictions, feed_dict={input_images: [test_images[i]]})[0]
    true_label = sess.run(true_labels, feed_dict={target_labels: [test_labels[i]]})[0]

    if predicted_label != true_label:
        misclassified_indices.append(i)

# Display a few misclassified images
num_images_to_display = 5
for i in range(num_images_to_display):
    index = misclassified_indices[i]
    image = test_images[index].reshape(28, 28)
    true_label = np.argmax(test_labels[index])
    predicted_label = sess.run(predictions, feed_dict={input_images: [test_images[index]]})[0]

    plt.imshow(image, cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    plt.show()
