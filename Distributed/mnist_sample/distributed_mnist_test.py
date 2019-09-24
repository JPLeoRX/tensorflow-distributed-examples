from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from mnist_shared import *

# Define and load datasets
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
dataset_train_raw = datasets['train']
dataset_test_raw = datasets['test']
NUM_OF_TRAIN_SAMPLES = info.splits['train'].num_examples
NUM_OF_TEST_SAMPLES = info.splits['test'].num_examples
print("{} samples in training dataset, {} samples in testing dataset".format(NUM_OF_TRAIN_SAMPLES, NUM_OF_TEST_SAMPLES))

# Define distributed strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_OF_WORKERS = strategy.num_replicas_in_sync
print("{} replicas in distribution".format(NUM_OF_WORKERS))

# Determine datasets sizes
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_OF_WORKERS

# Prepare training/testing dataset
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
dataset_train = dataset_train_raw.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).with_options(options)
dataset_test = dataset_test_raw.map(scale).batch(BATCH_SIZE).with_options(options)

# Show model summary, and evaluate it
model = load_model("model.h5")
model.summary()
eval_loss, eval_acc = model.evaluate(x=dataset_test)
print("")
print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))