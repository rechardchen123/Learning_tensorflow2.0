from CNNs.util import *

'''
Download the data
'''
flowers_url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
flowers_path = keras.utils.get_file("flowers.tgz", flowers_url, extract=True)
flowers_dir = os.path.join(os.path.dirname(flowers_path), "flower_photos")

for root, subdirs, files in os.walk(flowers_dir):
    print(root)
    for filename in files[:3]:
        print("   ", filename)
    if len(files) > 3:
        print("    ...")

'''
Build a keras.preprocessing.image.ImageDataGenerator that will preprocess the images and do some data augmentation (the documentation contains useful examples):
1. It should at least perform horizontal flips and keep 10% of the data for validation, but you may also make it perform a bit of rotation, rescaling, etc.
2. Also make sure to apply the Xception preprocessing function (using the preprocessing_function argument).
3. Call this generator's flow_from_directory() method to get an iterator that will load and preprocess the flower photos from the flower_photos directory, setting the target size to (299, 299) and subset to "training".
4. Call this method again with the same parameters except subset="validation" to get a second iterator for validation.
5. Get the next batch from the validation iterator and display the first image from the batch.
'''
datagen = keras.preprocessing.image.ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1,
    preprocessing_function=keras.applications.xception.preprocess_input)
train_generator = datagen.flow_from_directory(
    flowers_dir,
    target_size=(299,299),
    batch_size = 32,
    subset="training")

valid_generator = datagen.flow_from_directory(
    flowers_dir,
    target_size = (299,299),
    batch_size = 32,
    subset='validation')

X_batch, y_batch = next(valid_generator)
plt.imshow((X_batch[0] + 1)/2)
plt.axis("off")
plt.show()


'''
Now let's build the model:
1. Create a new Xception model, but this time set include_top=False to get the model without the top layer. Tip: you will need to access its input and output properties.
2. Make all its layers non-trainable.
3. Using the functional API, add a GlobalAveragePooling2D layer (feeding it the Xception model's output), and add a Dense layer with 5 neurons and the Softmax activation function.
4. Compile the model. Tip: don't forget to add the "accuracy" metric.
5. Fit your model using fit_generator(), passing it the training and validation iterators (and setting steps_per_epoch and validation_steps appropriately).
'''

n_classes = 5

base_model = keras.applications.xception.Xception(include_top=False)

for layer in base_model.layers:
    layer.trainable = False

global_pool = keras.layers.GlobalAveragePooling2D()(base_model.output)

predictions = keras.layers.Dense(n_classes, activation='softmax')(global_pool)

model = keras.models.Model(base_model.input, predictions)

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01), metrics=["accuracy"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=3306 // 32,
    epochs=50,
    validation_data=valid_generator,
    validation_steps=364 // 32)

pd.DataFrame(history.history).plot()
plt.axis([0, 19, 0, 1])
plt.show()


