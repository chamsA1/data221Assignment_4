from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
breast_cancer_data = load_breast_cancer()

# Feature matrix and target vector
feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target
feature_names = breast_cancer_data.feature_names

# Train-test split (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# Constrained Decision Tree (limit complexity)
constrained_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,              # <-- constraint to reduce overfitting
    random_state=42
)

# Train model
constrained_tree_model.fit(X_train, y_train)

# Predictions
train_predictions = constrained_tree_model.predict(X_train)
test_predictions = constrained_tree_model.predict(X_test)

# Accuracy
training_accuracy = accuracy_score(y_train, train_predictions)
testing_accuracy = accuracy_score(y_test, test_predictions)

print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)

# Feature importance
feature_importances = constrained_tree_model.feature_importances_

# Get indices of top 5 features
top_feature_indices = np.argsort(feature_importances)[-5:][::-1]

print("\nTop 5 Most Important Features:")
for index in top_feature_indices:
    print(f"{feature_names[index]}: {feature_importances[index]:.4f}")


# Limiting model complexity (e.g., max_depth) helps reduce overfitting
# by preventing the tree from memorizing the training data.
# This may lower training accuracy slightly but improves generalization.

# Feature importance shows which features most influence predictions.
# This helps interpret the model by identifying the most relevant variables.