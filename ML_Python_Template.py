# Loading the MNIST dataset in Keras


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

test_labels

# The network architecture

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# The compilation step

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Preparing the image data

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# "Fitting" the model**

model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Using the model to make predictions

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]

predictions[0].argmax()

predictions[0][7]

test_labels[0]