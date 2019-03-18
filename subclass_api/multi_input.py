from subclass_api.subclass_model import * 
import tensorflow as tf
from tensorflow import keras
'''
Now, if you want to send only features 0 to 4 directly to the output, and only features 2 to 7 through the hidden layers. 
tips:
1. You need to create two keras.layers.Input(input_A, input_B)
2. Build the model using the functional API, as above, but when you build the keras.models.Model, remember to set inputs =[input_A, input_B]
3. When calling fit(), evaluate() and predict(), instead of passing X_train_scaled, pass(X_train_scaled_A, X_train_scaled_B) 
'''
input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])

hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_A,hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs= [input_A,input_B],outputs = [output])
model.compile(loss="mean_squared_error", optimizer = "sgd")
model.summary()

X_train_scaled_A = X_train_scaled[:, :5]
X_train_scaled_B = X_train_scaled[:, 2:]
X_valid_scaled_A = X_valid_scaled[:, :5]
X_valid_scaled_B = X_valid_scaled[:, 2:]
X_test_scaled_A = X_test_scaled[:, :5]
X_test_scaled_B = X_test_scaled[:, 2:]

history = model.fit([X_train_scaled_A,X_train_scaled_B],y_train,epochs=10,
                    validation_data = ([X_valid_scaled_A,X_valid_scaled_B],y_valid))
model.evaluate([X_test_scaled_A, X_test_scaled_B], y_test)
model.predict([X_test_scaled_A,X_test_scaled_B])
