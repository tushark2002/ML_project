import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

if input("Do you want to train the model? (y/n): ") == "y":
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixel values to between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(
        y_test, 10)  # One-hot encode labels

    # Reshape the input data to make it suitable for an RNN
    # 28 time steps, each with 28 features
    x_train = x_train.reshape((x_train.shape[0], 28, 28))
    x_test = x_test.reshape((x_test.shape[0], 28, 28))

    # Build a simple Recurrent Neural Network (RNN) model
    model = models.Sequential([
        # Simple RNN layer with 64 units and ReLU activation
        layers.SimpleRNN(64, activation='relu', input_shape=(28, 28)),
        # Output layer with 10 neurons (for 10 classes) and softmax activation
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    model.save("./models/Recurrent-N-N.h5")
else:
    model = models.load_model(
        "./models/Recurrent-N-N.h5")

# Load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
