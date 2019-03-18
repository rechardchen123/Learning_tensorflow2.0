#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 3/18/2019 12:02 PM 
# @Author : Xiang Chen (Richard)
# @File : wide_and_deep_network.py 
# @Software: PyCharm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
''''
Use Keras' functional API to implement a Wide & Deep network to tackle the California housing problem.
Tips:
1. You need to create a keras.layers.Input layer to represent the inputs. Don't forget to specify the input shape.
2. Create the Dense layers, and connect them by using them like functions.
    For example, hidden1 = keras.layers.Dense(30, activation="relu")(input) and
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
3. Use the keras.layers.concatenate() function to concatenate the input layer and the second hidden layer's output.
4. Create a keras.models.Model and specify its inputs and outputs (e.g., inputs=[input]).
5. Then use this model just like a Sequential model:
    you need to compile it, display its summary, train it, evaluate it and use it to make predictions.
'''
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

input = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input) #connect with the input layer
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
concat = keras.layers.concatenate([input,hidden2]) #concatenate
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input],outputs = [output])
model.compile(loss="mean_squared_error", optimizer = 'sgd')
model.summary()
history = model.fit(X_train_scaled,y_train, epochs = 10,
                    validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
model.predict(X_test_scaled)

