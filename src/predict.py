import pandas as pd
import joblib
from src.preprocessing import preprocess_data
import os
from src.preprocessing import preprocess_data
MODEL_PATH = "outputs/models/decision_tree_model.pkl"
SAMPLE_PATH = "data/sample_data.csv"

def load_model(path=MODEL_PATH):
    """Load the trained model from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def predict_sample(model, sample_df):
    """Make a prediction on a single sample (as a DataFrame row)"""
    # Preprocess the sample the same way as training data
    X, _, _ = preprocess_data(sample_df)
    prediction = model.predict(X)
    return prediction[0]  # Return 0 or 1

if __name__ == "__main__":
    print("🚀 Loading model...")
    model = load_model()

    # Example: a single hardcoded sample (same structure as NSL-KDD, but only 1 row)
    sample_data = {
        "duration": [0],
        "protocol_type": ["tcp"],
        "service": ["http"],
        "flag": ["SF"],
        "src_bytes": [181],
        "dst_bytes": [5450],
        "land": [0],
        "wrong_fragment": [0],
        "urgent": [0],
        "hot": [0],
        "num_failed_logins": [0],
        "logged_in": [1],
        "num_compromised": [0],
        "root_shell": [0],
        "su_attempted": [0],
        "num_root": [0],
        "num_file_creations": [0],
        "num_shells": [0],
        "num_access_files": [0],
        "num_outbound_cmds": [0],
        "is_host_login": [0],
        "is_guest_login": [0],
        "count": [9],
        "srv_count": [9],
        "serror_rate": [0.00],
        "srv_serror_rate": [0.00],
        "rerror_rate": [0.00],
        "srv_rerror_rate": [0.00],
        "same_srv_rate": [1.00],
        "diff_srv_rate": [0.00],
        "srv_diff_host_rate": [0.00],
        "dst_host_count": [9],
        "dst_host_srv_count": [9],
        "dst_host_same_srv_rate": [1.00],
        "dst_host_diff_srv_rate": [0.00],
        "dst_host_same_src_port_rate": [1.00],
        "dst_host_srv_diff_host_rate": [0.00],
        "dst_host_serror_rate": [0.00],
        "dst_host_srv_serror_rate": [0.00],
        "dst_host_rerror_rate": [0.00],
        "dst_host_srv_rerror_rate": [0.00],
        "label": ["normal"],  # Only needed to satisfy preprocessing, not used in prediction
    }

    df_sample = pd.DataFrame(sample_data)

    print("🔎 Predicting...")
    result = predict_sample(model, df_sample)

    print(f"🔐 Prediction result: {'ATTACK 🚨' if result == 1 else 'NORMAL ✅'}")
