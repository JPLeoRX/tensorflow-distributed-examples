from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from Distributed.mnist_sample.mnist_shared import *

def main():
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

    # Build and train the model as multi worker
    with strategy.scope():
       model = build_and_compile_cnn_model()
    model.fit(x=dataset_train, epochs=1)

    # Show model summary, and evaluate it
    model.summary()
    eval_loss, eval_acc = model.evaluate(x=dataset_test)
    print("")
    print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

    # Save the model, reopen it and check that the state is preserved
    save_model(model, "model.h5")
    new_model = load_model("model.h5")
    predictions = model.predict(dataset_test)
    new_predictions = new_model.predict(dataset_test)
    np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)