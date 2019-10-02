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
        tf.keras.layers.Conv2D(48, 3, activation='relu', padding='same', input_shape=(96, 96, 3)),
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

        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Pre-compile it
    model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      metrics=['accuracy']
    )

    model.summary()

    return model