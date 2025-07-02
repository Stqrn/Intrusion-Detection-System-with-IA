import pandas as pd
from src.preprocessing import preprocess_data
from src.model import train_random_forest, evaluate_model, save_model
from src.visuals import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_class_distribution,
)

def main():
    print("📥 Loading dataset...")
    # CSV has no header → set header=None
    df = pd.read_csv("data/KDDTrain+.csv", header=None)

    print("⚙️ Preprocessing data...")
    X, y, _ = preprocess_data(df)

    print("✂️ Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("🌳 Training Random Forest model...")
    model = train_random_forest(X_train, y_train)

    print("📊 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("📈 Saving visual reports...")
    plot_confusion_matrix(y_test, model.predict(X_test))
    plot_classification_report(y_test, model.predict(X_test))
    plot_class_distribution(pd.concat([y_train, y_test]))

    print("💾 Saving model...")
    save_model(model, path="outputs/models/random_forest_model.pkl")

    print("✅ Done! Visual reports saved in: outputs/figures/")

if __name__ == "__main__":
    main()
