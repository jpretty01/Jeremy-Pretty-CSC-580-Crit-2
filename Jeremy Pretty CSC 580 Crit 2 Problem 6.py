# Jeremy Pretty
# CSC 580 Crit 2 Problem 6
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(101)
tf.set_random_seed(101)

# Generating random linear data
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)  # Number of data points

# 1) Plot the training data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data')
plt.show()

# 2) Create a TensorFlow model by defining placeholders X and Y
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 3) Declare two trainable TensorFlow variables for weights and bias
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 4) Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

# 5) Implement hypothesis, cost function, and optimizer
# Hypothesis: y_pred = W * X + b
y_pred = tf.add(tf.multiply(X, W), b)

# Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

# Optimizer: Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 6) Implement training process inside a TensorFlow session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

        # Print the current cost every 100 epochs
        if (epoch + 1) % 100 == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

# 7) Print the results for training cost, weight, and bias
print("Training cost =", training_cost, "Weight =", weight, "Bias =", bias)

# 8) Plot the fitted line on top of the original data
plt.scatter(x, y)
plt.plot(x, weight * x + bias, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Line')
plt.show()
