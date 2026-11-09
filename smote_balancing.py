import os
import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE

from config import ARTIFACTS_DIR, RANDOM_STATE


def apply_smote(X_train, y_train):
    """
    Apply SMOTE only to the training data.
    The test data is intentionally not used here.
    """

    print("\n⚖️ SMOTE BALANCING STARTED")

    print("\n📊 Class distribution BEFORE SMOTE:")
    print(y_train.value_counts().sort_index())

    smote = SMOTE(random_state=RANDOM_STATE)

    X_balanced, y_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    # Convert back to DataFrame/Series
    X_balanced = pd.DataFrame(
        X_balanced,
        columns=X_train.columns
    )

    y_balanced = pd.Series(
        y_balanced,
        name=y_train.name
    )

    print("\n✅ SMOTE Applied Successfully")

    print("\n📊 Class distribution AFTER SMOTE:")
    print(y_balanced.value_counts().sort_index())

    print("\n📐 Shape Information:")
    print("Original X_train:", X_train.shape)
    print("Balanced X_train:", X_balanced.shape)
    print("Original y_train:", y_train.shape)
    print("Balanced y_train:", y_balanced.shape)

    return X_balanced, y_balanced


def main():

    print("=" * 60)
    print("🚀 SMOTE BALANCING SCRIPT STARTED")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Check preprocessing outputs
    # --------------------------------------------------

    X_train_path = os.path.join(
        ARTIFACTS_DIR,
        "X_train.pkl"
    )

    y_train_path = os.path.join(
        ARTIFACTS_DIR,
        "y_train.pkl"
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
                "Please run preprocessing.py first."
            )

    # --------------------------------------------------
    # 2. Load training and testing data
    # --------------------------------------------------

    print("\n📥 Loading preprocessed data...")

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    print("✅ Preprocessed data loaded successfully")

    print("\nOriginal data:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # --------------------------------------------------
    # 3. Apply SMOTE ONLY to training data
    # --------------------------------------------------

    X_train_balanced, y_train_balanced = apply_smote(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 4. Save balanced training data
    # --------------------------------------------------

    X_balanced_path = os.path.join(
        ARTIFACTS_DIR,
        "X_train_balanced.pkl"
    )

    y_balanced_path = os.path.join(
        ARTIFACTS_DIR,
        "y_train_balanced.pkl"
    )

    joblib.dump(
        X_train_balanced,
        X_balanced_path
    )

    joblib.dump(
        y_train_balanced,
        y_balanced_path
    )

    print("\n💾 Balanced training data saved:")
    print("   ✓", X_balanced_path)
    print("   ✓", y_balanced_path)

    # --------------------------------------------------
    # 5. Verify test data remains unchanged
    # --------------------------------------------------

    print("\n🔍 TEST DATA CHECK")
    print("----------------------------")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nTest class distribution:")
    print(y_test.value_counts().sort_index())

    print("\n" + "=" * 60)
    print("✅ SMOTE BALANCING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()