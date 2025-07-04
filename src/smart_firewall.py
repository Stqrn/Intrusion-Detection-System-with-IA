from scapy.all import sniff, IP, TCP, UDP
import joblib
import pandas as pd
from datetime import datetime
import os

# Load trained model
model = joblib.load("outputs/models/best_model.pkl")

# Log file
os.makedirs("logs", exist_ok=True)
log_file = "logs/firewall_log.csv"

# تحويل الحزمة إلى ميزات (بسيطة كبداية)
def extract_features(packet):
    features = {
        "src_ip": packet[IP].src if IP in packet else "0.0.0.0",
        "dst_ip": packet[IP].dst if IP in packet else "0.0.0.0",
        "protocol": packet.proto if IP in packet else 0,
        "length": len(packet),
        "src_port": packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else 0,
        "dst_port": packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else 0,
    }
    return features

# تمرير الحزمة للنموذج
def predict_intrusion(features_dict):
    df = pd.DataFrame([features_dict])
    # مبدئيًا، نحذف الأعمدة غير الرقمية
    df = df.drop(["src_ip", "dst_ip"], axis=1)
    prediction = model.predict(df)
    return prediction[0]

# تسجيل النتيجة
def log_decision(features, result):
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()},{features['src_ip']},{features['dst_ip']},{features['protocol']},{result}\n")

# تنفيذ القرار
def handle_packet(packet):
    if IP not in packet:
        return

    features = extract_features(packet)
    result = predict_intrusion(features)

    if result == 1:
        print(f"🚨 Intrusion Detected from {features['src_ip']}")
        # يمكن إضافة أمر لمنع IP هنا
    else:
        print(f"✅ Allowed from {features['src_ip']}")

    log_decision(features, result)

# ابدأ الالتقاط
def start_firewall():
    print("🛡️ Smart Firewall running... Press CTRL+C to stop.")
    sniff(prn=handle_packet, store=0)

if __name__ == "__main__":
    start_firewall()
