from __future__ import absolute_import, division, print_function, unicode_literals

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper methods
#-----------------------------------------------------------------------------------------------------------------------
# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# Create neural network
def build_and_compile_cnn_model():
    # Declare model
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Pre-compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy']
    )

    return model
#-----------------------------------------------------------------------------------------------------------------------

# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["localhost:2222", "localhost:2223", "localhost:2224"]
    },
    'task': {'type': 'worker', 'index': 0}
})

# Define and load datasets
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
dataset_train_raw = datasets['train']
dataset_test_raw = datasets['test']

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print("Number of replicas in distribution: {}".format(strategy.num_replicas_in_sync))

# Determine datasets sizes
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
NUM_OF_WORKERS = strategy.num_replicas_in_sync
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS

# Prepare training dataset
dataset_train_unbatched = dataset_train_raw.map(scale).shuffle(BUFFER_SIZE)
dataset_train = dataset_train_unbatched.batch(BATCH_SIZE)

# Build and train the model as a single worker
#single_worker_model = build_and_compile_cnn_model()
#single_worker_model.fit(x=dataset_train, epochs=3)

# Build and train the model as multi worker
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=dataset_train, epochs=3)


exit()
