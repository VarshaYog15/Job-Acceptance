

import pandas as pd


RAW_DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/HR_Job_Placement_Dataset.csv"
CLEAN_DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/cleaned_data.csv"


def clean_data(df):
    print("\n🔍 Initial Data Shape:", df.shape)

    # --------------------------------------------------
    # 1️⃣ Remove Duplicates
    # --------------------------------------------------
    duplicate_count = df.duplicated().sum()
    print(f"🧹 Removing {duplicate_count} duplicate rows...")

    df = df.drop_duplicates().copy()

    # --------------------------------------------------
    # 2️⃣ Handle Missing Values
    # --------------------------------------------------
    num_cols = df.select_dtypes(include="number").columns
    df.loc[:, num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    print("✅ Missing values handled")

    # --------------------------------------------------
    # 3️⃣ Standardize Categorical Values
    # --------------------------------------------------
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).str.lower().str.strip()

    print("✅ Categorical values standardized")

    # --------------------------------------------------
    # 4️⃣ Fix Common Inconsistencies
    # --------------------------------------------------
    if "status" in df.columns:
        df.loc[:, "status"] = df["status"].replace({
            "placed ": "placed",
            "not placed ": "not placed",
            "placed.": "placed",
            "not_placed": "not placed"
        })

    if "gender" in df.columns:
        df.loc[:, "gender"] = df["gender"].replace({
            "m": "male",
            "f": "female"
        })

    print("✅ Inconsistent values corrected")

    # --------------------------------------------------
    # 5️⃣ Final Check
    # --------------------------------------------------
    print("\n📊 Final Data Shape:", df.shape)
    print("🧾 Total Missing Values Remaining:", df.isnull().sum().sum())

    return df


def main():
    print("🚀 SCRIPT STARTED")

    # Load data
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Clean data
    print("⚙️ Cleaning data...")
    df_cleaned = clean_data(df)

    # Show sample output
    print("\n📊 Sample Cleaned Data:")
    print(df_cleaned.head())

    # Save file
    df_cleaned.to_csv(CLEAN_DATA_PATH, index=False)

    print("\n💾 Cleaned data saved to:")
    print(CLEAN_DATA_PATH)

    print("\n✅ DATA CLEANING COMPLETED SUCCESSFULLY")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    main()

import pandas as pd


RAW_DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/HR_Job_Placement_Dataset.csv"
CLEAN_DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/cleaned_data.csv"


def clean_data(df):
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].str.lower().str.strip()

    return df


def main():
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("🧹 Cleaning data...")
    df_cleaned = clean_data(df)

    df_cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()

