import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def preprocess_data(df, target_col):
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return X_scaled, y, scaler, X_encoded.columns


def main():
    # ---------------------------
    # CONFIG
    # ---------------------------
    DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/job_acceptance_features.csv"
    TARGET_COL = "status"

    OUTPUT_DIR = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/artifacts"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # FETCH DATA
    # ---------------------------
    print("📥 Fetching data...")
    df = pd.read_csv(DATA_PATH)

    # ---------------------------
    # PREPROCESS DATA
    # ---------------------------
    print("⚙️ Preprocessing data...")
    X_scaled, y, scaler, feature_names = preprocess_data(df, TARGET_COL)

    # ---------------------------
    # SAVE OUTPUTS (PKL FILES)
    # ---------------------------
    joblib.dump(X_scaled, os.path.join(OUTPUT_DIR, "X_processed.pkl"))
    joblib.dump(y, os.path.join(OUTPUT_DIR, "y.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(OUTPUT_DIR, "feature_names.pkl"))

    print("✅ Outputs saved in:", OUTPUT_DIR)

    # ---------------------------
    # DISPLAY PKL CONTENTS
    # ---------------------------
    print("\n🔍 Loading PKL files for display...")

    X_loaded = joblib.load(os.path.join(OUTPUT_DIR, "X_processed.pkl"))
    y_loaded = joblib.load(os.path.join(OUTPUT_DIR, "y.pkl"))
    features_loaded = joblib.load(os.path.join(OUTPUT_DIR, "feature_names.pkl"))

    X_df = pd.DataFrame(X_loaded, columns=features_loaded)

    print("\n📊 Feature Matrix (first 5 rows):")
    print(X_df.head())

    print("\n🎯 Target Values (first 10):")
    print(y_loaded.head(10))

    print("\n📐 Shape Information:")
    print("X shape:", X_df.shape)
    print("y shape:", y_loaded.shape)

    print("\n🧾 Number of features:", len(features_loaded))
    print("🧾 First 10 feature names:")
    print(features_loaded[:10])


if __name__ == "__main__":
    main()
