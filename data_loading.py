import pandas as pd
from config import RAW_DATA_PATH


def load_data():
    """
    Load the raw job placement dataset.
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"Error: Dataset not found at:")
        print(RAW_DATA_PATH)
        raise

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise


if __name__ == "__main__":
    df = load_data()
    print("\nFirst 5 rows:")
    print(df.head())