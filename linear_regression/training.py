import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

tf.enable_eager_execution()
'''
the linear predict funciton can be describled as: y = w1*x1 + w2*x2 + w3*x3+w4*x4+w5*x5+w6*x6 + b.
The meaning of y is the delay time that we will predict. The w1 and x1 are the meaning of the weight and acutal_journeytime.
The rest of the parameters are the same meaning like w1 and x1. The b in  the rear of the equation is the bias. 
And the next is the cost function: we use the cost function to learn the weight and bias parameters.
Here we use the mean square euqation (MSE) to learn the weights and biases. The equation is :
MSE = (1/2n)*sum(y - (w1*x1 + w2*x2 + w3*x3+w4*x4+w5*x5+w6*x6 +b))^2.
And then, we use the gradient descend method to find the optimum values with iterations.
'''
class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()
        #here specify the shape of the tensors
        self.w = self.add_variable('weight',[6,1])
        self.b = self.add_variable('bias',[1])

        print(self.w.shape, self.b.shape)
        # print(type(self.w),tf.is_tensor(self.w),self.w.name)
        # print(type(self.b),tf.is_tensor(self.b),self.b.name)

    def call(self,x):
        x = tf.matmul(x,self.w) +self.b
        return x

def main():
    tf.set_random_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # assert tf.__version__.startswitch('2.')

    # load the data and then split the dataset into training dataset and test dataset
    data_address = '/home/richardchen123/Downloads/processed_data.csv'
    data = pd.read_csv(data_address)
    # get the training and test dataset
    X = data.iloc[:, 0:-1].astype(np.float32)
    y = data.iloc[:, -1].astype(np.float32)
    # split the data into X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # the shape of the dataset output: (2764,6),(2764,)(692,6),(692,)
    db_train = tf.data.Dataset.from_tensor_slices((X_train.values,y_train)).batch(8)
    db_val = tf.data.Dataset.from_tensor_slices((X_test.values,y_test)).batch(16)

    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

    for epoch in range(200):
        for step, (x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(epoch,'loss: ', loss.numpy())

        if epoch % 10 ==0:
            for x,y in db_val:
                logits = model(x)
                logits = tf.squeeze(logits, axis =1)
                loss = criteon(y, logits)

                print(epoch, 'val loss: ', loss.numpy())

if __name__=='__main__':
    main()