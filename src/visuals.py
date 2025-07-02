import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

# Save all figures to disk instead of showing
def save_fig(name):
    os.makedirs("outputs/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig("confusion_matrix")

def plot_classification_report(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.drop('accuracy', errors='ignore')

    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report.iloc[:, :3], annot=True, cmap='YlGnBu')
    plt.title('Classification Report Heatmap')
    save_fig("classification_report")

def plot_class_distribution(y):
    plt.figure(figsize=(5, 3))
    y.value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution (Original Data)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Normal', 'Attack'], rotation=0)
    save_fig("class_distribution")
