import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from xgboost import XGBClassifier

import joblib
import os

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear)": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

def main():
    print("📥 Loading dataset...")
    df = pd.read_csv("data/KDDTrain+.csv", header=None)
    print("⚙️ Preprocessing...")
    X, y, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"🧠 Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics


        if metrics["F1-score"] > best_score:
            best_score = metrics["F1-score"]
            best_model = model
            best_name = name


    df_results = pd.DataFrame(results).T
    print("\n📊 Results:\n", df_results)


    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x=df_results.index, y="F1-score", palette="Blues_d")
    plt.xticks(rotation=45)
    plt.title("Model Comparison (F1-score)")
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/model_comparison.png")
    print("📈 Chart saved to outputs/figures/model_comparison.png")


    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(best_model, "outputs/models/best_model.pkl")
    print(f"✅ Best model saved: {best_name} (F1: {best_score:.4f})")

if __name__ == "__main__":
    main()
