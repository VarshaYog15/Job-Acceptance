import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from config import (
    ARTIFACTS_DIR,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE
)


def train_random_forest(X_train, y_train):
    """
    Train the Random Forest model using the SMOTE-balanced
    training data.
    """

    print("\n🌲 RANDOM FOREST TRAINING STARTED")

    print("\n📊 Training data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\n🎯 Training class distribution:")
    print(y_train.value_counts().sort_index())

    # --------------------------------------------------
    # Create Random Forest model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\n⚙️ Training Random Forest...")
    
    model.fit(X_train, y_train)

    print("✅ Random Forest training completed")

    return model


def main():

    print("=" * 60)
    print("🚀 MODEL TRAINING SCRIPT STARTED")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------

    X_train_path = os.path.join(
        ARTIFACTS_DIR,
        "X_train_balanced.pkl"
    )

    y_train_path = os.path.join(
        ARTIFACTS_DIR,
        "y_train_balanced.pkl"
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
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"\nRequired file not found:\n{file_path}\n\n"
                "Please make sure preprocessing.py and "
                "smote_balancing.py have been executed successfully."
            )

    # --------------------------------------------------
    # 2. Load balanced training data
    # --------------------------------------------------

    print("\n📥 Loading balanced training data...")

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    print("✅ Balanced training data loaded")

    # --------------------------------------------------
    # 3. Load untouched test data
    # --------------------------------------------------

    print("\n📥 Loading test data...")

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    print("✅ Test data loaded")

    print("\n📐 Dataset Shapes")
    print("----------------------------")
    print("Balanced X_train:", X_train.shape)
    print("Balanced y_train:", y_train.shape)
    print("X_test          :", X_test.shape)
    print("y_test          :", y_test.shape)

    # --------------------------------------------------
    # 4. Train Random Forest
    # --------------------------------------------------

    model = train_random_forest(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 5. Save trained model
    # --------------------------------------------------

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n💾 Model saved successfully:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # 6. Display model information
    # --------------------------------------------------

    print("\n🌲 RANDOM FOREST INFORMATION")
    print("----------------------------")
    print("Number of trees:", model.n_estimators)
    print("Number of features:", model.n_features_in_)
    print("Classes:", model.classes_)

    # --------------------------------------------------
    # 7. Verify model can predict
    # --------------------------------------------------

    print("\n🔮 Testing model prediction...")

    sample_predictions = model.predict(
        X_test.iloc[:5]
    )

    print("First 5 predictions:")
    print(sample_predictions)

    sample_probabilities = model.predict_proba(
        X_test.iloc[:5]
    )

    print("\nFirst 5 prediction probabilities:")
    print(sample_probabilities)

    # --------------------------------------------------
    # 8. Feature importance preview
    # --------------------------------------------------

    feature_names_path = os.path.join(
        ARTIFACTS_DIR,
        "feature_names.pkl"
    )

    if os.path.exists(feature_names_path):

        feature_names = joblib.load(
            feature_names_path
        )

        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            by="importance",
            ascending=False
        )

        print("\n🔝 TOP 10 IMPORTANT FEATURES")
        print("----------------------------")
        print(
            feature_importance.head(10).to_string(index=False)
        )

    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()