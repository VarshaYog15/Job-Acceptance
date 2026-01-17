import pandas as pd
from config import RAW_DATA_PATH

def load_data():
    return pd.read_csv(RAW_DATA_PATH)
