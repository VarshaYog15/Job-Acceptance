import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from config import (
    ARTIFACTS_DIR,
    EVALUATION_OUTPUT_DIR
)


def load_model_and_test_data():
    """
    Load the already-trained model and untouched test data.
    """

    model_path = os.path.join(
        ARTIFACTS_DIR,
        "job_acceptance_model.pkl"
    )

    X_test_path = os.path.join(
        ARTIFACTS_DIR,
        "X_test.pkl"
    )

    y_test_path = os.path.join(
        ARTIFACTS_DIR,
        "y_test.pkl"
    )

    required_files = [
        model_path,
        X_test_path,
        y_test_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"\nRequired file not found:\n{file_path}\n\n"
                "Please run preprocessing.py, "
                "smote_balancing.py and model_training.py first."
            )

    print("\n📥 Loading trained model and test data...")

    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    print("✅ Model and test data loaded successfully")

    return model, X_test, y_test


def calculate_metrics(model, X_test, y_test):
    """
    Calculate classification metrics using untouched test data.
    """

    print("\n🔮 Generating predictions...")

    # Class predictions
    y_pred = model.predict(X_test)

    # Probability of class 1
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n📊 MODEL PERFORMANCE")
    print("=" * 45)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return (
        y_pred,
        y_probability,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )


def save_metrics(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
):

    metrics_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "model_metrics.txt"
    )

    with open(metrics_path, "w") as file:

        file.write("JOB ACCEPTANCE PREDICTION MODEL\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Accuracy : {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall   : {recall:.4f}\n")
        file.write(f"F1 Score : {f1:.4f}\n")
        file.write(f"ROC-AUC  : {roc_auc:.4f}\n")

    print("\n💾 Metrics saved:")
    print(metrics_path)


def save_classification_report(y_test, y_pred):

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Placed",
            "Placed"
        ],
        zero_division=0
    )

    report_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(
            "JOB ACCEPTANCE CLASSIFICATION REPORT\n"
        )
        file.write("=" * 45 + "\n\n")
        file.write(report)

    print("\n💾 Classification report saved:")
    print(report_path)

    print("\n📋 CLASSIFICATION REPORT")
    print("=" * 45)
    print(report)


def save_predictions(y_test, y_pred, y_probability):

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Placement_Probability": y_probability
    })

    results_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "model_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print("\n💾 Predictions saved:")
    print(results_path)


def plot_performance_metrics(
    accuracy,
    precision,
    recall,
    f1
):

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    values = [
        accuracy,
        precision,
        recall,
        f1
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values
    )

    plt.ylim(0, 1)

    plt.title(
        "Random Forest Model Performance"
    )

    plt.ylabel("Score")

    for i, value in enumerate(values):
        plt.text(
            i,
            value + 0.02,
            f"{value:.2f}",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "performance_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("\n📊 Performance chart saved:")
    print(output_path)


def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        ["Not Placed", "Placed"]
    )

    plt.yticks(
        [0, 1],
        ["Not Placed", "Placed"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    output_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("📊 Confusion matrix saved:")
    print(output_path)

    print("\nCONFUSION MATRIX")
    print("=" * 30)
    print(cm)


def plot_roc_curve(y_test, y_probability):

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Job Acceptance Prediction"
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("📈 ROC curve saved:")
    print(output_path)


def save_feature_importance(model):

    feature_names_path = os.path.join(
        ARTIFACTS_DIR,
        "feature_names.pkl"
    )

    if not os.path.exists(feature_names_path):

        print(
            "\n⚠️ feature_names.pkl not found. "
            "Skipping feature importance."
        )

        return

    feature_names = joblib.load(
        feature_names_path
    )

    importance_values = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    output_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        "feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False
    )

    print("\n💾 Feature importance saved:")
    print(output_path)

    print("\n🔝 TOP 10 IMPORTANT FEATURES")
    print("=" * 45)

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )


def main():

    print("=" * 60)
    print("🚀 MODEL EVALUATION STARTED")
    print("=" * 60)

    os.makedirs(
        EVALUATION_OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. Load model and untouched test data
    # --------------------------------------------------

    model, X_test, y_test = load_model_and_test_data()

    print("\n📐 Test data:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------
    # 2. Calculate metrics
    # --------------------------------------------------

    (
        y_pred,
        y_probability,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ) = calculate_metrics(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------
    # 3. Save metrics
    # --------------------------------------------------

    save_metrics(
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )

    # --------------------------------------------------
    # 4. Save classification report
    # --------------------------------------------------

    save_classification_report(
        y_test,
        y_pred
    )

    # --------------------------------------------------
    # 5. Save predictions
    # --------------------------------------------------

    save_predictions(
        y_test,
        y_pred,
        y_probability
    )

    # --------------------------------------------------
    # 6. Performance visualization
    # --------------------------------------------------

    plot_performance_metrics(
        accuracy,
        precision,
        recall,
        f1
    )

    # --------------------------------------------------
    # 7. Confusion matrix
    # --------------------------------------------------

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------
    # 8. ROC curve
    # --------------------------------------------------

    plot_roc_curve(
        y_test,
        y_probability
    )

    # --------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------

    save_feature_importance(
        model
    )

    print("\n" + "=" * 60)
    print("🎉 MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()