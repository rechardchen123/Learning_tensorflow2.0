from Loading_and_preprocessing_Data.util import *
tf.enable_eager_execution()

filename_dataset = tf.data.Dataset.list_files(train_filenames)

'''
Parse a CSV line:
1. create a parse_csv_line() function that takes a single line as argument.
2. call tf.io.decode_csv() to parse that line.
3. call tf.stack() to create a single tensor containing all the input features(i.e., all fields except the last one)
4. reshape the labels filed(i.e., the last field) to give it a shape of [1] instead of [](i.e., it must not be a scalar). You can use tf.reshape(label_field,[1]),or call tf.stack([label_field]),or use label_field[yf.newaxis].
5. Return a tuple with both tensors(input features and labels).
6. try calling it on a single line from one of the CSV files.
'''
n_inputs = X_train.shape[1]

def parse_csv_line(line, n_inputs = n_inputs):
    defs = [tf.constant(np.nan)]*(n_inputs+1)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return x, y

'''
Now, create a csv_reader_dataset() function that takes a list of CSV filenames and returns a dataset that will provide batches of parsed and shuffled data from these files, including the features and labels, repeating the whole data once per epoch.
'''
def csv_reader_dataset(filenames, n_parse_threads=5, batch_size=32, shuffle_buffer_size=10000, n_readers=5):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,num_parallel_calls= n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

'''
Building a training set, valida set and a test set using your csv_reader_dataset() function
'''
batch_size = 32
train_set = csv_reader_dataset(train_filenames,batch_size)
valid_set = csv_reader_dataset(valid_filenames,batch_size)
test_set = csv_reader_dataset(test_filenames,batch_size)

'''
Build and compile a Keras model for this regression task, and use your datasets to train it, evaluate it and make predictions for the test one.
'''
model = keras.models.Sequential(
    [keras.layers.Dense(30, activation='relu',input_shape=[n_inputs]),
    keras.layers.Dense(1)])

model.compile(loss="mse",optimizer=tf.train.AdamOptimizer(learning_rate=0.005))
model.fit(train_set, steps_per_epoch=len(X_train)//batch_size,epochs=10, validation_data=valid_set, validation_steps=len(X_valid)//batch_size)

model.evaluate(test_set, steps=len(X_test)//batch_size)

new_set = test_set.map(lambda X,y:X)
model.predict(new_set, steps=len(X_test)//batch_size)









