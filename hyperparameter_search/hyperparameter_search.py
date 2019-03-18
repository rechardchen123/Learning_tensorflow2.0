#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 3/18/2019 11:43 AM 
# @Author : Xiang Chen (Richard)
# @File : hyperparameter_search.py 
# @Software: VS code
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


housing = fetch_california_housing()
print(housing.DESCR)
housing.data.shape
housing.target.shape

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
len(X_train), len(X_valid), len(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.transform(X_test)

'''
Try training your model multiple times, with different a learning rate each time 
(e.g., 1e-4, 3e-4, 1e-3, 3e-3, 3e-2), and compare the learning curves. 
For this, you need to create a keras.optimizers.SGD optimizer and specify the learning_rate in its constructor, 
then pass this SGD instance to the compile() method using the optimizer argument.
'''
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for learning_rate in learning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    callbacks = [keras.callbacks.EarlyStopping(patience=10)]
    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_test_scaled, y_test), epochs=100,
                        callbacks=callbacks)
    histories.append(history)



for learning_rate, history in zip(learning_rates, histories):
    print("learning rate: ", learning_rate)
    plot_learning_curves(history)

