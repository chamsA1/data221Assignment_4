from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
breast_cancer_data = load_breast_cancer()

# Feature matrix and target vector
feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

# Perform 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# Initialize Decision Tree classifier using entropy
decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train the model
decision_tree_model.fit(X_train, y_train)

# Generate predictions
train_predictions = decision_tree_model.predict(X_train)
test_predictions = decision_tree_model.predict(X_test)

# Compute accuracy
training_accuracy = accuracy_score(y_train, train_predictions)
testing_accuracy = accuracy_score(y_test, test_predictions)

print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)


# Entropy measures the level of uncertainty or impurity in the target variable.
# A value of 0 means the data is perfectly pure (all samples belong to one class),
# while higher values indicate more mixed classes.
# Decision tree use entropy to choose splits that maximize information gain,
# meaning they reduce uncertainty as much as possible at each step.
#
# The training accuracy is very high, while the testing accuracy
# is slightly lower. This means that the model is likely overfitting,
# as it learns the training data too well and does not generalize perfectly
# to unseen data.