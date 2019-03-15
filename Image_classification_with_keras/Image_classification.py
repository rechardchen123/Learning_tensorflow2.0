#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 3/15/2019 10:33 AM 
# @Author : Xiang Chen (Richard)
# @File : Image_classification.py 
# @Software: PyCharm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.preprocessing import StandardScaler

# load the data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

print(X_train.shape)
print(X_train[0])

plt.imshow(X_train[0], cmap='binary')
plt.show()

print(y_train)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

n_rows = 5
n_cols = 10
plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap='binary', interpolation='nearest')
        plt.axis('off')
        plt.title(class_names[y_train[index]])
plt.show()

'''
Build a classification neural network with keras

'''
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()  #examine the output
keras.utils.plot_model(model, 'my_mnist_model.png',show_shapes=True) #generate an image of model's architecture

#after the model is created, you must call its compile() method to specify the loss function and the optmizer to use.
# in this case, you want to use the "sparse_categorical_crossentropy" loss, and the "sgd" optimizer(stochastic gradient
# descent). Moreover, you can optionally specify a list of additional metrics that should be measured during training.
# In this case you should specify metrics=["accuracy"]. Note: you can find more loss functions in keras.losses,
# more metrics in keras.metrics and more optimizers in keras.optimizers.
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

#Now your model is ready to be trained. Call its fit() method, passing it the input features (X_train) and
# the target classes (y_train). Set epochs=10 (or else it will just run for a single epoch).
# You can also (optionally) pass the validation data by setting validation_data=(X_valid, y_valid).
# If you do, Keras will compute the loss and the additional metrics (the accuracy in this case) on
# the validation set at the end of each epoch. If the performance on the training set is much better than
# on the validation set, your model is probably overfitting the training set
# (or there is a bug, such as a mismatch between the training set and the validation set).
# Note: the fit() method will return a History object containing training stats.
# Make sure to preserve it (history = model.fit(...)).
history = model.fit(X_train, y_train, epochs = 10, validation_data=(X_valid, y_valid))

#Try running pd.DataFrame(history.history).plot() to plot the learning curves.
# To make the graph more readable, you can also set figsize=(8, 5), call plt.grid(True) and plt.gca().set_ylim(0, 1).
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

#Call the model's evaluate() method, passing it the test set (X_test and y_test).
# This will compute the loss (cross-entropy) on the test set,
# as well as all the additional metrics (in this case, the accuracy).
# Your model should achieve over 80% accuracy on the test set.
model.evaluate(X_test, y_test)

#When using Gradient Descent, it is usually best to ensure that the features all have a similar scale,
# preferably with a Normal distribution. Try to standardize the pixel values and see if this improves
# the performance of your neural network.
pixel_means = X_train.mean(axis=0)
pixel_stds = X_train.std(axis=0)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
X_valid_scaled = scaler.transform(X_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_test_scaled = scaler.transform(X_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train_scaled, y_train, epochs=20,
                    validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
plot_learning_curves(history)












