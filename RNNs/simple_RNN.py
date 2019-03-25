from RNNs.util import *

'''
Let's create a simple 2-layer RNN with 100 neurons per layer, plus a dense layer with a single neuron:
'''
input_shape = X_train_3D.shape[1:]

model1 = keras.models.Sequential()
model1.add(keras.layers.SimpleRNN(100, return_sequences=True, input_shape=input_shape))
model1.add(keras.layers.SimpleRNN(50))
model1.add(keras.layers.Dense(1))
model1.compile(loss='mse', optimizer = keras.optimizers.SGD(lr=0.005),metrics=["mae"])

history1 = model1.fit(X_train_3D, y_train, epochs=200, batch_size=200, validation_data=(X_valid_3D,y_valid))

'''
plot the history
'''
def plot_history(history,loss='loss'):
    train_losses = hisotry.history[loss]
    valid_losses = history.history["val_" + loss]
    n_epochs = len(hisotry.epoch)
    minloss = np.min(valid_losses)

    plt.plot(train_losses, color='b', label='Train')
    plt.plot(valid_losses,color='r',label='validation')
    plt.plot([0, n_epochs],[minloss,minloss],'k--',label="Min val: {:.2f}".format(minloss))
    plt.axis([0, n_epochs, 0 ,20])
    plt.legend()
    plt.show()

plot_history(history1)

'''
evaluate the model
'''
model1.evaluate(X_valid_3D, y_valid)

def huber_loss(y_true, y_pred, max_grad=1.):
    err = tf.abs(y_true-y_pred,name='abs')
    mg = tf.constant(max_grad,name='max_grad')
    lin = mg * (err - 0.5 * mg)
    quad = 0.5 * err *err
    return tf.where(err < mg, quad, lin)

model1.evaluate(X_valid_3D, y_valid)

'''
Plot the predictions
'''
y_pred_rnn1 = model1.predit(X_valid_3D)
plot_predictions(("Target", y_valid),
                 ("Linear", y_pred_linear),
                 ("RNN", y_pred_rnn1),
                 end=365)


