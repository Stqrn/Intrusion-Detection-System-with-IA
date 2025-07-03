from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree Classifier"""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data"""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

def save_model(model, path="outputs/models/decision_tree_model.pkl"):
    """Save the trained model to disk"""
    joblib.dump(model, path)
    print(f"📦 Model saved to {path}")

def load_model(path="outputs/models/random_forest_model.pkl"):
    """Load a trained model from file."""
    return joblib.load(path)
