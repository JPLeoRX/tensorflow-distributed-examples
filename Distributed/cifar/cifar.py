import pickle
import numpy as np
import tensorflow as tf

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    # print("x = {}".format(x))
    # encoded = np.zeros((len(x), 10))
    # for idx, val in enumerate(x):
    #     encoded[idx][val] = 1
    # return encoded
    return x

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode, np.array(test_features), np.array(test_labels), 'preprocess_training.p')

def build_and_compile_cnn_model():
    # Declare model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(48, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(48, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(96, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(96, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Pre-compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      metrics=['accuracy']
    )

    model.summary()

    return model
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



# Preprocess all the data and save it (run this only the first time)
print('Loading, pre-processing and saving raw dataset')
preprocess_and_save_data('dataset', normalize, one_hot_encode)

# Load the saved dataset batches, and save them as joined set
print('Loading pre-processed dataset, and join 5 batches')
batch1_features, batch1_labels = pickle.load(open('preprocess_batch_1.p', mode='rb'))
batch2_features, batch2_labels = pickle.load(open('preprocess_batch_2.p', mode='rb'))
batch3_features, batch3_labels = pickle.load(open('preprocess_batch_3.p', mode='rb'))
batch4_features, batch4_labels = pickle.load(open('preprocess_batch_4.p', mode='rb'))
batch5_features, batch5_labels = pickle.load(open('preprocess_batch_5.p', mode='rb'))
batch_all_features = np.concatenate([batch1_features, batch2_features, batch3_features, batch4_features, batch5_features])
batch_all_labels = np.concatenate([batch1_labels, batch2_labels, batch3_labels, batch4_labels, batch5_labels])
print("Pre-processed batch 1 set shape: {}".format(batch1_features.shape))
print("Pre-processed batch 2 set shape: {}".format(batch2_features.shape))
print("Pre-processed batch 3 set shape: {}".format(batch3_features.shape))
print("Pre-processed batch 4 set shape: {}".format(batch4_features.shape))
print("Pre-processed batch 5 set shape: {}".format(batch5_features.shape))
print("Pre-processed batch all set shape: {}".format(batch_all_features.shape))
pickle.dump((batch_all_features, batch_all_labels), open('preprocess_batch_all.p', 'wb'))

# Load dataset
print('Loading pre-processed training/testing/validation dataset')
training_features, training_labels = pickle.load(open('preprocess_batch_all.p', mode='rb'))
testing_features, testing_labels = pickle.load(open('preprocess_training.p', mode='rb'))
validation_features, validation_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
print("Pre-processed training set shape: {}, {}".format(training_features.shape, training_labels.shape))
print("Pre-processed testing set shape: {}".format(testing_features.shape))
print("Pre-processed validation set shape: {}".format(validation_features.shape))

# Convert pre-processed data to tensorflow datasets
print('Building tensorflow datasets from pre-processed dataset')
train_dataset_unbatched = tf.data.Dataset.from_tensor_slices((training_features, training_labels))
test_dataset_unbatched = tf.data.Dataset.from_tensor_slices((testing_features, testing_labels))
validation_dataset_unbatched = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))

# Define distributed strategy
print('Defined tensorflow strategy')
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets sizes, and prepare training/testing dataset
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
train_dataset = train_dataset_unbatched.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).with_options(options)
test_dataset = test_dataset_unbatched.batch(BATCH_SIZE).with_options(options)

# Build and train the model as multi worker
with strategy.scope():
    model = build_and_compile_cnn_model()
model.fit(x=train_dataset, epochs=5)

# Show model summary, and evaluate it
model.summary()
eval_loss, eval_acc = model.evaluate(x=test_dataset)
print("")
print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

# Save the model, reopen it and check that the state is preserved
model.save("model.h5")
new_model = tf.keras.models.load_model("model.h5")
predictions = model.predict(test_dataset)
new_predictions = new_model.predict(test_dataset)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-5, atol=1e-5)