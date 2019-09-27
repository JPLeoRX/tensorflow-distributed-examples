import tensorflow as tf
import horovod.tensorflow.keras as hvd

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
      tf.keras.layers.Dense(4096, activation='relu'),
      tf.keras.layers.Dense(2048, activation='relu'),
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    # Pre-compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=opt,
      metrics=['accuracy']
    )

    return model

# Save model to file
def save_model(model, path):
    model.save(path)

# Load model from file
def load_model(path):
    return tf.keras.models.load_model(path)

# Save model to file (using SavedModel)
def save_model_with_saved_model(model, path):
    tf.keras.experimental.export_saved_model(model, path)

# Load model from file (using SavedModel)
def load_model_with_saved_model(path):
    return tf.keras.experimental.load_from_saved_model(path)