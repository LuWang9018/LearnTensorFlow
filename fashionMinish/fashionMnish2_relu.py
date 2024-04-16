"""Beyond Hello World, A Computer Vision Example"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

labelDict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

number_of_classes = training_labels.max() + 1

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy"
)

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)


model.predict(training_images[0:10])

data_idx = 59999  # The question number to study with. Feel free to change up to 59999.

plt.figure()
plt.imshow(training_images[data_idx], cmap="gray")
plt.colorbar()
plt.grid(False)
plt.show()

x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(training_images[data_idx : data_idx + 1]).flatten())
plt.xticks(range(10))
plt.show()

print(
    "correct answer:",
    training_labels[data_idx],
    ":",
    labelDict[training_labels[data_idx]],
)
