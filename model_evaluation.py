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
