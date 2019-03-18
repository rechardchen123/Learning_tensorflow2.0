#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 3/15/2019 2:34 PM 
# @Author : Xiang Chen (Richard)
# @File : neural net for regression.py 
# @Software: PyCharm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
'''
Load the California housing dataset using sklearn.datasets.fetch_california_housing. 
This returns an object with a DESCR attribute describing the dataset, a data attribute with the input features, 
and a target attribute with the labels. The goal is to predict the price of houses in a district (a census block) 
given some stats about that district. This is a regression task (predicting values).
'''
housing = fetch_california_housing()
print(housing.DESCR)
housing.data.shape
housing.target.shape

'''
Split the dataset into a training set, a validation set and a test set using Scikit-Learn's 
sklearn.model_selection.train_test_split() function.
'''
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
len(X_train), len(X_valid), len(X_test)

'''
Scale the input features (e.g., using a sklearn.preprocessing.StandardScaler). 
Once again, don't forget that you should not fit the validation set or the test set, only the training set.
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.transform(X_test)

'''
Now build, train and evaluate a neural network to tackle this problem. Then use it to make predictions on the test set.
'''
model = keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer = 'sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=10)]
history = model.fit(X_train_scaled,y_train,
                    validation_data=(X_test_scaled,y_test),epochs=100,
                    callbacks=callbacks)
model.evaluate(X_test_scaled,y_test)
model.predict(X_test_scaled)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)


