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

'''
Load CIFAR10 using keras.datasets.cifar10.load_data(), and split it into a training set (45,000 images), a validation set (5,000 images) and a test set(10,000 images). Make sure that pixel values range from 0 to 1. Visualize a few images using plt.imshow().
'''
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",]

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train_full[:-5000] / 255
y_train = y_train_full[:-5000]
X_valid = X_train_full[-5000:] / 255
y_valid = y_train_full[-5000:]
X_test = X_test / 255

plt.figure (figsize=(10,7))
n_rows, n_cols = 10, 15
for row in range(n_rows):
    for col in range(n_cols):
        i = row * n_cols + col
        plt.subplot(n_rows, n_cols, i +1)
        plt.axis("off")
        plt.imshow(X_train[i])

for i in range(n_cols):
    print(classes[y_train[i][0]], end=" ")




