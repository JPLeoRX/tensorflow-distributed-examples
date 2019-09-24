import tensorflow as tf

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