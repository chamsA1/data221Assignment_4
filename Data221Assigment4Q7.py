import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Data221Assignment4Q6 import cnn_model, X_test_images, y_test_labels

# Generate predictions
predicted_probabilities = cnn_model.predict(X_test_images)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
plt.title("CNN Confusion Matrix")
plt.show()

# --- Find misclassified indices ---
misclassified_indices = np.where(predicted_labels != y_test_labels)[0]

# Select first 3 misclassified examples
sample_misclassified = misclassified_indices[:3]

# Class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Visualize misclassified images
plt.figure(figsize=(10, 4))

for i, index in enumerate(sample_misclassified):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test_images[index].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_test_labels[index]]}\nPred: {class_names[predicted_labels[index]]}")
    plt.axis('off')

plt.suptitle("Misclassified Images")
plt.show()

# Misclassifications often occur between visually similar categories
# such as shirts, pullovers, and coats.

# One way to improve performance is to use a deeper CNN or add more
# convolutional layers to capture more complex features.