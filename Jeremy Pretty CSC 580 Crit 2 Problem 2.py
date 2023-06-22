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
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Define the loss function and optimizer
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the neural network
model.fit(train_images, train_labels, epochs=20, batch_size=32)

# Find some misclassified images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
misclassified_indices = np.where(predicted_labels != true_labels)[0]

# Display a few misclassified images
num_images_to_display = 5
for i in range(num_images_to_display):
    index = misclassified_indices[i]
    image = test_images[index].reshape(28, 28)
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]

    plt.imshow(image, cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    plt.show()
