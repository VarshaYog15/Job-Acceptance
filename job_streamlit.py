import streamlit as st
import pandas as pd
from config import CLEAN_DATA_PATH

st.set_page_config(page_title="Job Acceptance Predictor")

st.title("Job Acceptance EDA & Predictor")

@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA_PATH)

df = load_data()

st.success("Dataset loaded successfully")
st.write("Preview of data:")
st.dataframe(df.head())
