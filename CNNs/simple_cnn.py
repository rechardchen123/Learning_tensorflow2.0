from CNNs.util import * 

'''
Build and train a baseline model with a few dense layers, and plot the learning curves. Use the model's summary() method to count the number of parameters in this model.
Tip: recall that to plot the learning curves, you can simply create a Pandas DtaFrame with history.history dict, then call its plot() method.
'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32,32,3]),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(10, activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy',
optimizer = keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=20,
validation_data=(X_valid,y_valid))

pd.DataFrame(history.history).plot()
plt.axis([0,19,0,1])
plt.show()

model.summary()
