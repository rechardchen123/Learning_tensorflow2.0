import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
Let's go back to Fashion MNIST and build deep nets to tackle it. We need to load it, split it and scale it.
'''


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full),(X_test,y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000],X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
X_valid_scaled = scaler.transform(X_valid.astype(np.float32).reshape(
    -1, 1)).reshape(-1, 28, 28)
X_test_scaled = scaler.transform(X_test.astype(np.float32).reshape(
    -1, 1)).reshape(-1, 28, 28)

'''
Build a sequential model with 20 hidden dense layers, with 100 neurons each, using the ReLU activation function, plus the output layer(10 neurons, softmax activation function).
'''
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
#    model.add(keras.layers.Dense(100))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(100,activation = 'selu',kernel_initializer='lecun_normal'))
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train_scaled,y_train,epochs=10, validation_data=(X_valid_scaled,y_valid))
plot_learning_curves(history)
model.summary()

