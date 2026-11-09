import pandas as pd

from config import RAW_DATA_PATH, CLEAN_DATA_PATH


def clean_data(df):
    """
    Clean the raw job placement dataset.
    """

    print("\n🔍 Initial Data Shape:", df.shape)

    # ==========================================
    # 1. REMOVE DUPLICATES
    # ==========================================

    duplicate_count = df.duplicated().sum()

    print(f"🧹 Removing {duplicate_count} duplicate rows...")

    df = df.drop_duplicates().copy()

    # ==========================================
    # 2. HANDLE MISSING VALUES
    # ==========================================

    # Numerical columns → median
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())

    # Categorical columns → mode
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        if not df[col].mode().empty:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    print("✅ Missing values handled")

    # ==========================================
    # 3. STANDARDIZE CATEGORICAL VALUES
    # ==========================================

    for col in cat_cols:
        df.loc[:, col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    print("✅ Categorical values standardized")

    # ==========================================
    # 4. CORRECT INCONSISTENT LABELS
    # ==========================================

    # Status
    if "status" in df.columns:

        df.loc[:, "status"] = df["status"].replace({
            "placed": "Placed",
            "not placed": "Not Placed",
            "not_placed": "Not Placed",
            "placed.": "Placed",
            "not placed.": "Not Placed"
        })

    # Gender
    if "gender" in df.columns:

        df.loc[:, "gender"] = df["gender"].replace({
            "m": "male",
            "f": "female"
        })

    print("✅ Inconsistent values corrected")

    # ==========================================
    # 5. FINAL MISSING-VALUE CHECK
    # ==========================================

    remaining_missing = df.isnull().sum().sum()

    print("\n📊 Final Data Shape:", df.shape)
    print("🧾 Total Missing Values Remaining:", remaining_missing)

    return df


def main():

    print("🚀 DATA CLEANING SCRIPT STARTED")

    # ==========================================
    # LOAD RAW DATA
    # ==========================================

    print("\n📥 Loading raw data...")

    try:
        df = pd.read_csv(RAW_DATA_PATH)

    except FileNotFoundError:

        print("\n❌ ERROR: Raw dataset not found.")
        print(f"Expected location:\n{RAW_DATA_PATH}")
        raise

    print("✅ Raw dataset loaded successfully.")

    # ==========================================
    # CLEAN DATA
    # ==========================================

    print("\n⚙️ Cleaning data...")

    df_cleaned = clean_data(df)

    # ==========================================
    # DISPLAY SAMPLE
    # ==========================================

    print("\n📊 Sample Cleaned Data:")
    print(df_cleaned.head())

    # ==========================================
    # SAVE CLEANED DATA
    # ==========================================

    df_cleaned.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    print("\n💾 Cleaned data saved to:")
    print(CLEAN_DATA_PATH)

    print("\n✅ DATA CLEANING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()