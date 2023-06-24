import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(101)
tf.random.set_seed(101)

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

# 2) Define the model architecture using Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 3) Define hyperparameters
learning_rate = 0.001
training_epochs = 10000

# 4) Compile the model with a loss function and optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate),
              loss='mean_squared_error')

# 5) Perform data preprocessing
x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)

# 6) Train the model
history = model.fit(x_norm, y_norm, epochs=training_epochs, verbose=0)

# 7) Print the results for training loss, weight, and bias
training_loss = history.history['loss'][-1]
weight, bias = model.get_weights()
print("Training loss =", training_loss)
print("Weight =", weight)
print("Bias =", bias)

# 8) Plot the fitted line on top of the original data
plt.scatter(x, y)
plt.plot(x, np.squeeze(weight) * x_norm + bias, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Line')
plt.show()
