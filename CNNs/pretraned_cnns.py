from CNNs.util import *

'''
Using keras.preprocessing.image.load_img() followed by keras.preprocessing.image.img_to_array(), load one or more images (e.g., fig.jpg or ostrich.jpg in the images folder). You should set target_size=(299, 299) when calling load_img(), as this is the shape that the Xception network expects.
'''
img_fig_path = os.path.join("images","fig.jpg")
img_fig = keras.preprocessing.image.load_img(
    img_fig_path, target_size=(299, 299))
img_fig = keras.preprocessing.image.img_to_array(img_fig)

plt.imshow(img_fig / 255)
plt.axis("off")
plt.show()
img_fig.shape

img_ostrich_path = os.path.join("images", "ostrich.jpg")
img_ostrich = keras.preprocessing.image.load_img(
    img_ostrich_path, target_size=(299, 299))
img_ostrich = keras.preprocessing.image.img_to_array(img_ostrich)


plt.imshow(img_ostrich / 255)
plt.axis("off")
plt.show()
img_ostrich.shape

'''
Create a batch containing the images you just loaded, and preprocrss thsis batch using keras.applications.xception.process_input(). Verify that the features now vary from -1 to 1: this is what the Xception network expects.
'''
X_batch = np.array([img_fig, img_ostrich])
X_preproc = keras.applications.xception.preprocess_input(X_batch)
X_preproc.min(), X_preproc.max()

'''
Create an instance of the Xception model (keras.applications.xception.Xception) and use its predict() method to classify the images in the batch. You can use keras.applications.resnet50.decode_predictions() to convert the output matrix into a list of top-N predictions (with their corresponding class labels).
'''
model = keras.applications.xception.Xception()
Y_proba = model.predict(X_preproc)

Y_proba.shape

np.argmax(Y_proba, axis=1)

decoded_predictions = keras.applications.resnet50.decode_predictions(Y_proba)
for preds in decoded_predictions:
    for wordnet_id, name, proba in preds:
        print("{} ({}): {:.1f}%".format(name, wordnet_id, 100 * proba))
    print()


