import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
(X_train_images, y_train_labels), (X_test_images, y_test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
X_train_images = X_train_images / 255.0
X_test_images = X_test_images / 255.0

# Reshape to include channel dimension (28, 28, 1)
X_train_images = X_train_images.reshape(-1, 28, 28, 1)
X_test_images = X_test_images.reshape(-1, 28, 28, 1)

# Build CNN model
cnn_model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=10, activation='softmax')  # 10 classes
])

# Compile model
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model (at least 15 epochs)
cnn_model.fit(X_train_images, y_train_labels, epochs=15, batch_size=32, validation_split=0.1)

# Evaluate on test set
test_loss, test_accuracy = cnn_model.evaluate(X_test_images, y_test_labels)

print("Test Accuracy:", test_accuracy)

# CNNs are preferred for image data because they preserve spatial structure
# and learn local patterns, unlike fully connected networks.

# The convolution layer learns features such as edges, textures,
# and shapes that help distinguish different clothing items.