#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
'''
The subclassing API:
1. create a subclass of the keras.models.Model class
2. create all the layers you need in the constructor (e.g. self.hidden1 = keras.layers.Dense(...)).
3. Use the layers to process the input in the call() method, and return the output.
4. Note that you do not need to create a keras.layers.Input in this case.
5. Also note that self.output is used by keras, so you should use another name for the output layer(e.g. self.output_layer).
'''


class MyModel(keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden1 = keras.layers.Dense(30, activation='relu')
        self.hidden2 = keras.layers.Dense(30, activation='relu')
        self.output_ = keras.layers.Dense(1)

    def call(self, input):
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_(concat)
        return output


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


housing = fetch_california_housing()
print(housing.DESCR)
housing.data.shape
housing.target.shape

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
len(X_train), len(X_valid), len(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = MyModel()
model.compile(loss='mse',optimizer='sgd')
history = model.fit(X_train_scaled,y_train,epochs=10,
                    validation_data = (X_valid_scaled,y_valid))
model.summary()
model.evaluate(X_test_scaled,y_test)
model.predict(X_test_scaled)

'''
Now, if you want to send only features 0 to 4 directly to the output, and only features 2 to 7 through the hidden layers. 
'''
input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])
