<<<<<<< HEAD
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

INPUT_FILE = "model_results.csv"
OUTPUT_DIR = "evaluation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)
    return df["Actual"], df["Predicted"]


def plot_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = [accuracy, precision, recall, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    plt.figure()
    plt.bar(labels, metrics)
    plt.title("Model Performance Metrics")
    plt.savefig(f"{OUTPUT_DIR}/performance_metrics.png")
    plt.close()

    print("📊 Saved → performance_metrics.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()

    print("📊 Saved → confusion_matrix.png")


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.close()

    print("📊 Saved → roc_curve.png")


def save_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)

    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    print("💾 Saved → classification_report.txt")


def main():
    print("\n📊 MODEL EVALUATION STARTED\n")

    y_true, y_pred = load_data()

    save_report(y_true, y_pred)
    plot_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)

    print("\n🎉 EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
=======
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(X, y, model_path):
    """
    Train RandomForest model and evaluate it
    """

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # Model training
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("🤖 Training model...")
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    print("\n📊 Model Evaluation (Test Data)")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # Save model
    # ---------------------------
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved at: {model_path}")

    return model


def main():
    # ---------------------------
    # CONFIG
    # ---------------------------
    ARTIFACTS_DIR = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/artifacts"
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, "job_acceptance_model.pkl")

    # ---------------------------
    # LOAD PREPROCESSED DATA
    # ---------------------------
    print("📥 Loading preprocessed data...")
    X = joblib.load(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
    y = joblib.load(os.path.join(ARTIFACTS_DIR, "y.pkl"))

    print("✅ Data loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ---------------------------
    # TRAIN + EVALUATE
    # ---------------------------
    train_and_evaluate(X, y, MODEL_PATH)


if __name__ == "__main__":
    main()
>>>>>>> 4a72a78348968a5fc2040d881e72091356356487
