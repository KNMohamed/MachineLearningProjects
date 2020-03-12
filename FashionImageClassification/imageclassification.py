# @Author: Khalid
# TensorFlow
import tensorflow as tf

# High level API for tensorflow
from tensorflow import keras

# Helper libraries
import numpy as np
# For graphing
import matplotlib.pyplot as plt


# Define function to plot an image
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]), color=color)


# Define a function to plot a barchart with the output probabilities
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Load fashion data-set
data = keras.datasets.fashion_mnist

# Split data into testing and training data
# 60,000 images to train data, 10,000 images to test accuracy
# Dataset returns four NumPy arrays
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Scale values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Flattens 28x28 pixel images
    keras.layers.Dense(128, activation='relu'), # rectified linear unit is the most popular activation function for deep neural networks
    keras.layers.Dense(10,activation="softmax"), # last layer of a classification network interpreted as probability distribution
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)

# uncomment to test model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Acc:", test_acc)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction = probability_model.predict(test_images)





num_rows = 5
num_cols = 3
numimages = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(numimages):
    plt.subplot(num_rows, 2* num_cols, 2 * i + 1)
    plot_image(i, prediction[i], test_labels, test_images)
    plt.subplot(num_rows, 2* num_cols, 2 * i + 2)
    plot_value_array(i, prediction[i], test_labels)
plt.tight_layout()
plt.show()