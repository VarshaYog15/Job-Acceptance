import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from config import (
    FEATURE_DATA_PATH,
    ARTIFACTS_DIR,
    FEATURE_NAMES_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    RANDOM_STATE
)


def preprocess_data(df):
    """
    Preprocess the feature-engineered dataset.

    Steps:
    1. Separate target variable
    2. Convert target labels to 0/1
    3. Split into training and testing sets
    4. Identify numerical and categorical columns
    5. Fit preprocessing only on training data
    6. Transform training and testing data
    """

    print("\n🔧 PREPROCESSING STARTED")

    df = df.copy()

    # --------------------------------------------------
    # 1. Check target column
    # --------------------------------------------------
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    # --------------------------------------------------
    # 2. Separate features and target
    # --------------------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    print("\n🎯 Target column:", TARGET_COLUMN)

    # --------------------------------------------------
    # 3. Convert target to numeric
    # --------------------------------------------------
    if y.dtype == "object":
        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "placed": TARGET_MAPPING["Placed"],
                "not placed": TARGET_MAPPING["Not Placed"],
                "not_placed": TARGET_MAPPING["Not Placed"]
            })
        )

    if y.isnull().any():
        raise ValueError(
            "Target column contains values that could not be converted "
            "to 0/1. Check the status labels."
        )

    y = y.astype(int)

    print("\n📊 Target Distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------
    # 4. Train-Test Split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n✂️ Train-Test Split Completed")
    print("Training samples:", X_train.shape[0])
    print("Testing samples :", X_test.shape)

    # --------------------------------------------------
    # 5. Identify numerical and categorical columns
    # --------------------------------------------------
    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\n🔢 Numerical Features:")
    print(numerical_columns)

    print("\n🔤 Categorical Features:")
    print(categorical_columns)

    # --------------------------------------------------
    # 6. Create preprocessing pipeline
    # --------------------------------------------------
    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False
                ),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------
    # 7. FIT ONLY ON TRAINING DATA
    # --------------------------------------------------
    print("\n⚙️ Fitting preprocessing on training data...")

    X_train_processed = preprocessor.fit_transform(X_train)

    # --------------------------------------------------
    # 8. Transform test data
    # --------------------------------------------------
    X_test_processed = preprocessor.transform(X_test)

    # Convert to DataFrame
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index
    )

    feature_names = X_train_processed.columns.tolist()

    print("\n✅ Preprocessing Completed")
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape :", X_test_processed.shape)
    print("Total processed features:", len(feature_names))

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names
    )


def main():

    print("=" * 60)
    print("🚀 PREPROCESSING SCRIPT STARTED")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load feature-engineered data
    # --------------------------------------------------
    print("\n📥 Loading feature-engineered data...")

    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"Feature-engineered dataset not found:\n"
            f"{FEATURE_DATA_PATH}\n\n"
            f"Please run feature_engineering.py first."
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    print("✅ Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    # --------------------------------------------------
    # 2. Preprocess
    # --------------------------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        feature_names
    ) = preprocess_data(df)

    # --------------------------------------------------
    # 3. Save processed datasets
    # --------------------------------------------------
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    train_path = os.path.join(
        ARTIFACTS_DIR,
        "X_train.pkl"
    )

    test_path = os.path.join(
        ARTIFACTS_DIR,
        "X_test.pkl"
    )

    y_train_path = os.path.join(
        ARTIFACTS_DIR,
        "y_train.pkl"
    )

    y_test_path = os.path.join(
        ARTIFACTS_DIR,
        "y_test.pkl"
    )

    preprocessor_path = os.path.join(
        ARTIFACTS_DIR,
        "preprocessor.pkl"
    )

    joblib.dump(X_train, train_path)
    joblib.dump(X_test, test_path)
    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    joblib.dump(
        feature_names,
        FEATURE_NAMES_PATH
    )

    # --------------------------------------------------
    # 4. Display results
    # --------------------------------------------------
    print("\n💾 Files saved successfully:")
    print("   ✓", train_path)
    print("   ✓", test_path)
    print("   ✓", y_train_path)
    print("   ✓", y_test_path)
    print("   ✓", preprocessor_path)
    print("   ✓", FEATURE_NAMES_PATH)

    print("\n📊 FINAL SHAPES")
    print("----------------------------")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\n🧾 First 10 processed features:")
    for feature in feature_names[:10]:
        print("   •", feature)

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()