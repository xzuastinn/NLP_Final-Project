# evaluation.py
# ------------------------------------------
# Central evaluation module to measure model performance.
# ------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print classification metrics and show a confusion matrix heatmap.
    """
    print(f"\nEvaluation Report for {model_name}:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["positive", "neutral", "negative"], yticklabels=["positive", "neutral", "negative"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()


# utils.py
# ------------------------------------------
# Helper utilities for plotting, mapping, etc.
# ------------------------------------------
def label_to_int(label):
    """Convert string label to integer for neural net training."""
    mapping = {"positive": 0, "neutral": 1, "negative": 2}
    return mapping[label]

def int_to_label(index):
    """Convert integer index back to label."""
    mapping = {0: "positive", 1: "neutral", 2: "negative"}
    return mapping[index]
