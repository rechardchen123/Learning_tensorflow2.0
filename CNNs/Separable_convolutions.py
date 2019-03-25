from CNNs.util import *

model = keras.models.Sequential([
    kears.layers.Conv2D(filters=32, kernel_size=3, padding='same',activation='relu',input_shape=[32,32,3]),
    keras.layers.BatchNormalization(),
    keras.layers.SeparableCon2D(filters=32, kernel_size=3, padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.SeparableConv2D(
        filters=64, kernel_size=3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.SeparableConv2D(
        filters=64, kernel_size=3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01), metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot()
plt.axis([0, 19, 0, 1])
plt.show()
model.summary()


