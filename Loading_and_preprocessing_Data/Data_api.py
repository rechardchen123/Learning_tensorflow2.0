from util import *
tf.enable_eager_execution()

filename_dataset = tf.data.Dataset.list_files(train_filenames)
for filename in filename_dataset:
    print(filename)
    