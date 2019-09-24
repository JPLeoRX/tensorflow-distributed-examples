from __future__ import absolute_import, division, print_function, unicode_literals

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from mnist_shared import scale, build_and_compile_cnn_model

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
print("{} samples in training dataset. {} samples in testing dataset".format(num_train_examples, num_test_examples))

# Prepare training/testing dataset
dataset_train_unbatched = dataset_train_raw.map(scale).shuffle(BUFFER_SIZE)
dataset_train = dataset_train_unbatched.batch(BATCH_SIZE)
dataset_test = dataset_test_raw.map(scale).batch(BATCH_SIZE)

# Build and train the model as a single worker
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(x=dataset_train, epochs=3)
eval_loss, eval_acc = single_worker_model.evaluate(x=dataset_test)
print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))


# Build and train the model as multi worker
#with strategy.scope():
#  multi_worker_model = build_and_compile_cnn_model()
#multi_worker_model.fit(x=dataset_train, epochs=3)
#multi_worker_model.evaluate(x=dataset_test)


exit()
