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
