import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Smart Firewall Dashboard", layout="wide")

# Load logs
def load_data(path="logs/firewall_log.csv"):
    try:
        df = pd.read_csv(path, names=["time", "src_ip", "dst_ip", "protocol", "result"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except FileNotFoundError:
        st.error("❌ Log file not found.")
        return pd.DataFrame()

df = load_data()

st.title("🛡️ Smart Firewall Dashboard")

if df.empty:
    st.warning("No data to display.")
else:
    # Filters
    st.sidebar.title("🔍 Filters")
    ip_filter = st.sidebar.text_input("🔎 Filter by Source IP:")

    if ip_filter:
        df = df[df["src_ip"].str.contains(ip_filter)]

    st.subheader("📋 Latest Intrusions")
    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("📈 Intrusions Over Time")
    df["hour"] = df["time"].dt.floor("h")
    intrusions_per_hour = df[df["result"] == 1].groupby("hour").size()

    fig1, ax1 = plt.subplots()
    intrusions_per_hour.plot(ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("🚨 Intrusions Per Hour")
    st.pyplot(fig1)

    st.subheader("🌐 Top Source IPs")
    top_ips = df[df["result"] == 1]["src_ip"].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_ips.values, y=top_ips.index, ax=ax2)
    ax2.set_xlabel("Number of Attacks")
    ax2.set_title("Top 10 Source IPs")
    st.pyplot(fig2)

    st.subheader("📊 Protocol Usage")
    proto_counts = df[df["result"] == 1]["protocol"].value_counts()

    fig3, ax3 = plt.subplots()
    proto_counts.plot.pie(autopct='%1.1f%%', ax=ax3)
    ax3.set_ylabel("")
    ax3.set_title("Protocols Used in Attacks")
    st.pyplot(fig3)
