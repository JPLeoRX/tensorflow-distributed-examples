from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd
from mnist_shared import *

def main():
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())\
    tf.keras.backend.set_session(tf.Session(config=config))

    # Define and load datasets
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    dataset_train_raw = datasets['train']
    dataset_test_raw = datasets['test']
    NUM_OF_TRAIN_SAMPLES = info.splits['train'].num_examples
    NUM_OF_TEST_SAMPLES = info.splits['test'].num_examples
    print("{} samples in training dataset, {} samples in testing dataset".format(NUM_OF_TRAIN_SAMPLES, NUM_OF_TEST_SAMPLES))

    # Define distributed strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
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

    callbacks = [
       # Horovod: broadcast initial variable states from rank 0 to all other processes.
       # This is necessary to ensure consistent initialization of all workers when
       # training is started with random weights or restored from a checkpoint.
       hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
       callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    model.fit(x=dataset_train, epochs=10, callbacks=callbacks)

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