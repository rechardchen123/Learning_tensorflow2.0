from Loading_and_preprocessing_Data.util import *
from Loading_and_preprocessing_Data.Data_api import *
'''
Write a csv_to_tfrecords() function that will read from a given csv dataset(e.g., such as train_set, passed as an argument), and write the instances to multiple TFRecord files. The number of files should be defined by an n_shards argument. If there are 20 shards, then the files should be named my_train_00000-to-00019.tfrecords to my_train_00019-to-00019.tfrecords, where the my_train prefix should be defined by an argument.
'''


def serialize_example(x, y):
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(feature={
        "input_features": tf.train.Feature(float_list=input_features),
        "label": tf.train.Feature(float_list=label)})
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def csv_to_tfrecords(filename, csv_reader_dataset, n_shards, steps_per_shard, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    for shard in range(n_shards):
        path = "{}_{:05d}-of-{:05d}.tfrecords".format(
            filename, shard, n_shards)
        with tf.io.TFRecordWriter(path, options) as writer:
            for X_batch, y_batch in csv_reader_dataset.take(steps_per_shard):
                for x_instance, y_instance in zip(X_batch, y_batch):
                    writer.write(serialize_example(x_instance, y_instance))


'''
Use this function to write the training set, validation set and test set to multiple TFRecord files.
'''
batch_size = 32
n_shards = 20
steps_per_shard = len(X_train)//batch_size//n_shards
csv_to_tfrecords("my_train.tfrecords", train_set, n_shards, steps_per_shard)

n_shards = 1
steps_per_shard = len(X_valid) // batch_size // n_shards
csv_to_tfrecords("my_valid.tfrecords", valid_set, n_shards, steps_per_shard)

n_shards = 1
steps_per_shard = len(X_test) // batch_size // n_shards
csv_to_tfrecords("my_test.tfrecords", test_set, n_shards, steps_per_shard)

'''
write a tfrecords_reader_dataset() function, very similar to csv_reader_dataset(), that will read from multiple TFRecord files. For convenience, it should take a file prefix(such as "my_train") and use os.listdir() to look for all the TFRecord files with that prefix.
'''
expected_features = {
    "input_features": tf.io.FixedLenFeature([n_inputs], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32)}


def parse_tfrecord(serialized_example):
    example = tf.io.parse_single_example(serialize_example, expected_features)
    return example["input_features"], example["label"]


def tfrecords_reader_dataset(filename, batch_size=32, shuffle_buffer_size=10000, n_reader=5):
    filenames = [name for name in os.listdir() if name.startswith(
        filename) and name.endswith(".tfrecords")]
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename), cycle_length=n_reader)
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(parse_tfrecord,
                                                               batch_size,
                                                               num_parallel_calls=tf.data.experimental.AUTOTUNE))
    return dataset.prefetch(1)


tfrecords_train_set = tfrecords_reader_dataset("my_train", batch_size=3)

for X_batch, y_batch in tfrecords_train_set.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

batch_size = 32
tfrecords_train_set = tfrecords_reader_dataset("my_train", batch_size)
tfrecords_valid_set = tfrecords_reader_dataset("my_valid", batch_size)
tfrecords_test_set = tfrecords_reader_dataset("my_test", batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[n_inputs]),
    keras.layers.Dense(1),
])

model.compile(loss="mse", optimizer=tf.train.AdamOptimizer(
    learning_rate=0.005))

model.fit(tfrecords_train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=tfrecords_valid_set, validation_steps=len(X_valid) // batch_size)

model.evaluate(tfrecords_test_set, steps=len(X_test) // batch_size)

new_set = test_set.map(lambda X, y: X)
model.predict(new_set, steps=len(X_test) // batch_size)
