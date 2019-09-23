from __future__ import absolute_import, division, print_function, unicode_literals

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

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
dataset_train = datasets['train']
dataset_test = datasets['test']

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Determine sizes
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(strategy.num_replicas_in_sync)

exit()




# Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label