from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
breast_cancer_data = load_breast_cancer()

# Features and labels
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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network model (1 hidden layer, sigmoid output is default for binary)
neural_network_model = MLPClassifier(
    hidden_layer_sizes=(16,),   # one hidden layer with 16 neurons
    activation='relu',
    max_iter=500,
    random_state=42
)

# Train model
neural_network_model.fit(X_train_scaled, y_train)

# Predictions
train_predictions = neural_network_model.predict(X_train_scaled)
test_predictions = neural_network_model.predict(X_test_scaled)

# Accuracy
training_accuracy = accuracy_score(y_train, train_predictions)
testing_accuracy = accuracy_score(y_test, test_predictions)

print("Training Accuracy:", training_accuracy)
print("Testing Accuracy:", testing_accuracy)


# Feature scaling is necessary because neural networks are sensitive
# to the scale of input features, and scaling helps the model converge faster.

# An epoch is one complete pass through the entire training dataset
# during the learning process.