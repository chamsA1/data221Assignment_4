from sklearn.datasets import load_breast_cancer
import numpy as np

# Load dataset
data = load_breast_cancer()

# Feature matrix and target vector
X = data.data
y = data.target

# Shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Count samples in each class
unique, counts = np.unique(y, return_counts=True)

for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

# Class labels meaning
print("Target names:", data.target_names)

# The dataset is  imbalanced because there are more benign (357) samples than malignant (212) samples.
# Class balance is important because many machine learning models can become
# biased toward the majority class. This can lead to misleading accuracy,
# where the model performs well overall but poorly on the minority class.
# In medical datasets like this one, correctly identifying the minority
# class (malignant tumors) is especially critical, since misclassification
# could have serious consequences.