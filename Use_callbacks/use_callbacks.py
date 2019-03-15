#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 3/15/2019 2:06 PM 
# @Author : Xiang Chen (Richard)
# @File : use_callbacks.py 
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

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# The fit() method accepts a callbacks argument. Try training your model with a large number of epochs,
# a validation set, and with a few callbacks from keras.callbacks:
# TensorBoard: specify a log directory. It should be a subdirectory of a root logdir,
# such as ./my_logs/run_1, and it should be different every time you train your model.
# You can use a timestamp in the subdirectory's path to ensure that it changes at every run.
# EarlyStopping: specify patience=5
# ModelCheckpoint: specify the path of the checkpoint file to save
# (e.g., "my_mnist_model.h5") and set save_best_only=True
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])
root_logdir = r'C:\Users\LPT-ucesxc0\Documents\Github-repositories\Learning_tensorflow2.0\Use_callbacks'
logdir = os.path.join(root_logdir, "run{}".format(time.time()))
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.EarlyStopping(patience=5),
    keras.callbacks.ModelCheckpoint("my_mnist_model.h5",save_best_only=True),
]
pixel_means = X_train.mean(axis=0)
pixel_stds = X_train.std(axis=0)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

history = model.fit(X_train_scaled,y_train, epochs=50,
                    validation_data=(X_valid_scaled,y_valid),
                    callbacks=callbacks)
model = keras.models.load_model("my_mnist_model.h5")
model.evaluate(X_valid_scaled, y_valid)

