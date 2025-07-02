import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Features to encode
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

def preprocess_data(df):
    # If the file has no header, add the correct column names
    df.columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty_level"
    ]

    # Drop unnecessary column
    df.drop("difficulty_level", axis=1, inplace=True)

    # Label binary classification: 'normal' → 0, others → 1 (attack)
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc  # Keep for inverse transform if needed

    # Separate features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y, encoders
