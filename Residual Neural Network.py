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

    # Reshape the input data to make it suitable for a Convolutional Neural Network (CNN)
    # Add a channel dimension
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    # Add a channel dimension
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Build a simplified Residual Neural Network (ResNet) model

    def resnet_block(x, filters, kernel_size=3, stride=1):
        y = layers.Conv2D(filters, kernel_size=kernel_size,
                        strides=stride, padding='same', activation='relu')(x)
        y = layers.Conv2D(filters, kernel_size=kernel_size,
                        strides=stride, padding='same')(y)
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, kernel_size=1,
                            strides=stride, padding='same')(x)
        return layers.add([x, y])

    input_tensor = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(input_tensor)
    x = resnet_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = resnet_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = resnet_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    model.save("./models/Residual-N-N.h5")
else:
    model = models.load_model(
        "./models/Residual-N-N.h5")

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
