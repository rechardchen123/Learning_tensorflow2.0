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
Save/load a SavedModel
'''
tf.enable_eager_execution()

(X_train_full, y_train_full),(X_test,y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.
X_test = X_test/255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
y_train = y_train.astype(np.int64)
y_valid = y_valid.astype(np.int64)
y_test = y_test.astype(np.int64)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.train.AdamOptimizer(learning_rate=0.005),metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

MODEL_NAME = "my_fashion_mnist"

model_version = int(time.time())
model_path = os.path.join(MODEL_NAME,str(model_version))
os.makedirs(model_path)

tf.saved_model.save(model, model_path)

for root, dirs, files in os.walk(MODEL_NAME):
    indent = "    " * root.count(os.sep)
    print("{}{}/".format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + '   ',filename))


