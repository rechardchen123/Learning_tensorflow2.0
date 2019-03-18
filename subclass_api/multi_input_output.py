from subclass_api.multi_input import *
import tensorflow as tf
from tensorflow import keras
'''
There are many use cases in which having multiple outputs can be useful:
1. Your task may require multiple outputs, for example, you may want to locate and classify the main object in a picture. This is both a regression task(finding the coordinates of the object's center, as well as its width and hegiht) and a classifcation task.
2. Similarly, you may have multiple indepedent tasks to perform based on the same data. Sure, you could train one neural network per task, but in many cases you will get better results on all tasks by training a single neural network with one output per task. This is becasue the neural network can learn features in the data that are useful across tasks.
3. Another use case is as a regularization technique (i.e., a training constraint whose objective is to reduce overfitting and thus improve the model's ability to generalize).For example, you may want to add some auxiliary outputs in a neural network architecture to ensure that that the underlying part of the network learns something useful on its own, without relying on the rest of the network.
Tips:
1. Building the model is pretty straightforward using the functional API. Just make sure you specify both outputs when creating the keras.models.Model, for example, outpts = [output, aux_output].
2. Each output has its own loss function. In this scenario, they will be identical, so you can either specify loss='mse' or loss = ['mse','mse'], which does the same thing.
3. The final loss used to train the whole network is just a weighted sum of all loss functions. In this scenario, you want most to give a much smaller weight to the auxiliary output, so when compiling the model, you must specify loss_weights=[0.9,0.1].
4. When calling fit () or evaluate(), you need to pass the labels for all outputs. In this scenario the lanels will be the same for the main output and for the auxiliary output, so make sure to pass (y_train, y_train) instead of y_train.
5. The predict() method will return both the main output and the auxiliary output.
'''
input_A = keras.layers.Input (shape=X_train_scaled_A.shape[1:])
input_B = keras.layers.Input(shape=X_test_scaled_B.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2)

model = keras.models.Model(inputs=[input_A,input_B],
outputs = [output, aux_output])

model.compile(
    loss="mean_squared_error", loss_weights=[0.9, 0.1], optimizer="sgd")

model.summary()
history = model.fit([X_train_scaled_A, X_train_scaled_B], [y_train, y_train],
                    epochs=10,
                    validation_data=([X_valid_scaled_A, X_valid_scaled_B],
                    [y_valid, y_valid]))
model.evaluate([X_test_scaled_A, X_test_scaled_B],[y_test, y_test])
y_pred, y_pred_aux = model.predict([X_test_scaled_A, X_test_scaled_B])
y_pred
y_pred_aux
