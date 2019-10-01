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

        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
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