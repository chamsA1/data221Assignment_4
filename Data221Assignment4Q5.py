from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix,
    target_vector,
    test_size=0.2,
    random_state=42,
    stratify=target_vector
)

# --- Decision Tree (constrained) ---
constrained_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

constrained_tree_model.fit(X_train, y_train)

# --- Neural Network ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neural_network_model = MLPClassifier(
    hidden_layer_sizes=(16,),
    max_iter=500,
    random_state=42
)

neural_network_model.fit(X_train_scaled, y_train)

# --- Decision Tree Confusion Matrix ---
decision_tree_predictions = constrained_tree_model.predict(X_test)

decision_tree_conf_matrix = confusion_matrix(y_test, decision_tree_predictions)

print("Decision Tree Confusion Matrix:")
print(decision_tree_conf_matrix)

ConfusionMatrixDisplay(confusion_matrix=decision_tree_conf_matrix).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()


# --- Neural Network Confusion Matrix ---
neural_network_predictions = neural_network_model.predict(X_test_scaled)

neural_network_conf_matrix = confusion_matrix(y_test, neural_network_predictions)

print("\nNeural Network Confusion Matrix:")
print(neural_network_conf_matrix)

ConfusionMatrixDisplay(confusion_matrix=neural_network_conf_matrix).plot()
plt.title("Neural Network Confusion Matrix")
plt.show()

# The neural network is preferred because it usually achieves slightly
# higher accuracy and better overall performance.

# Decision Tree:
# Advantage: Easy to interpret and understand.
# Limitation: Prone to overfitting if not properly constrained.

# Neural Network:
# Advantage: Can capture more complex patterns in the data.
# Limitation: Less interpretable and requires more tuning.