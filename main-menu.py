import os
from src.model import load_model, evaluate_model
from src.visuals import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_class_distribution,
)
from src.preprocessing import preprocess_data
from src.model import train_random_forest, save_model
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_PATH = "outputs/models/random_forest_model.pkl"
DATA_PATH = "data/KDDTrain+.csv"

def train():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH, header=None)
    X, y, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("🌳 Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    print("📊 Evaluating model...")
    evaluate_model(model, X_test, y_test)
    print("📈 Saving visual reports...")
    from pandas import concat
    plot_confusion_matrix(y_test, model.predict(X_test))
    plot_classification_report(y_test, model.predict(X_test))
    plot_class_distribution(concat([y_train, y_test]))
    save_model(model, MODEL_PATH)
    print("✅ Model trained and saved.")

def test_interactive():
    try:
        model = load_model(MODEL_PATH)
    except:
        print("❌ No trained model found. Please train the model first.")
        return
    from src.predict_interactive import predict_interactive
    predict_interactive(model)

def show_menu():
    while True:
        print("\n=== Intrusion Detection System Menu ===")
        print("1. 🧠 Train Model")
        print("2. 🔍 Test Model (Interactive)")
        print("3. 📊 View Graphs (Saved in outputs/figures)")
        print("4. ❌ Exit")
        choice = input("Select an option (1-4): ")

        if choice == "1":
            train()
        elif choice == "2":
            test_interactive()
        elif choice == "3":
            print("📂 Open outputs/figures/ to view the graphs.")
        elif choice == "4":
            print("👋 Exiting...")
            break
        else:
            print("⚠️ Invalid choice. Try again.")

if __name__ == "__main__":
    show_menu()
