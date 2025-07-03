import pandas as pd
from src.preprocessing import preprocess_data
from src.model import load_model

def predict_interactive(model=None):
    print("\n=== 🔍 Interactive Prediction ===")
    print("You can choose to:")
    print("1. Enter input manually")
    print("2. Load input from a CSV file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        predict_from_input(model)
    elif choice == "2":
        predict_from_csv(model)
    else:
        print("⚠️ Invalid choice.")

def predict_from_input(model):
    print("\nEnter values for the following features (numeric only):")
    columns = [
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
        "dst_host_srv_rerror_rate"
    ]

    input_data = []
    for col in columns:
        val = input(f"{col}: ")
        try:
            input_data.append(float(val))
        except ValueError:
            print("⚠️ Invalid input. Please enter numeric values only.")
            return

    df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(df)[0]
    print("🔒 Intrusion detected!" if prediction == 1 else "✅ Normal connection.")

def predict_from_csv(model):
    path = input("Enter path to CSV file: ")
    if not path.endswith(".csv"):
        print("⚠️ Please provide a valid CSV file.")
        return

    try:
        df = pd.read_csv(path, header=None)
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
        df.drop(["label", "difficulty_level"], axis=1, inplace=True)

        X, _, _ = preprocess_data(df)
        predictions = model.predict(X)

        for i, pred in enumerate(predictions):
            result = "🔒 Intrusion" if pred == 1 else "✅ Normal"
            print(f"Row {i+1}: {result}")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
