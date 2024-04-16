"""Beyond Hello World, A Computer Vision Example"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

# setting values to rows and column variables
rows = 4
columns = 2

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

# create figure
fig = plt.figure(figsize=(5, 7))

for i in range(8):
    fig.add_subplot(rows, columns, i + 1)
    # showing image
    plt.imshow(training_images[i])
    plt.axis("off")
    plt.title("{}:{}".format(training_labels[i], labelDict[training_labels[i]]))

    # Adds a subplot at the 2nd position

data_idx = 6174

plt.figure()
plt.imshow(test_images[data_idx], cmap="gray")
plt.colorbar()
plt.grid(False)
plt.show()
