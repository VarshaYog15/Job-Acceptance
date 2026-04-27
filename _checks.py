import pandas as pd


def initial_data_checks(df):
    print("\n📊 DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n🧾 DATA TYPES")
    print(df.dtypes)

    print("\n🔍 SAMPLE RECORDS")
    print(df.head())

    print("\n🚨 NULL VALUE DISTRIBUTION")
    nulls = df.isna().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "No missing values found")

    print("\n🔁 DUPLICATE RECORDS")
    print("Duplicate rows:", df.duplicated().sum())


def main():
    DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/job_acceptance_features.csv"

    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    initial_data_checks(df)


if __name__ == "__main__":
    main()
