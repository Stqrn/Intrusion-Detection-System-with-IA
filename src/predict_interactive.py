import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from src.preprocessing import preprocess_data

MODEL_PATH = "outputs/models/random_forest_model.pkl"

# Features we'll ask the user to enter manually
MANUAL_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "logged_in", "count", "srv_count"
]

DEFAULT_VALUES = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 100,
    "dst_bytes": 500,
    "logged_in": 1,
    "count": 5,
    "srv_count": 5,
    "label": "normal"  # required by preprocess function
}

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

def get_user_input():
    print("🧠 Enter values for a new network connection sample:")
    sample = {}
    for feature in MANUAL_FEATURES:
        val = input(f"{feature} [{DEFAULT_VALUES[feature]}]: ").strip()
        sample[feature] = val if val else DEFAULT_VALUES[feature]

    # Convert numerical fields
    for key in ["duration", "src_bytes", "dst_bytes", "logged_in", "count", "srv_count"]:
        sample[key] = int(sample[key])

    sample["label"] = "normal"  # dummy label for compatibility
    return pd.DataFrame([sample])

def predict_sample(model, df_sample):
    X, _, _ = preprocess_data(df_sample)
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    print("🚀 Loading model...")
    model = load_model()

    sample_df = get_user_input()

    print("🔍 Predicting...")
    result = predict_sample(model, sample_df)

    print(f"\n🔐 Prediction result: {'ATTACK 🚨' if result == 1 else 'NORMAL ✅'}")
