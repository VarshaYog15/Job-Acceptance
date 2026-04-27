<<<<<<< HEAD
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
=======
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
>>>>>>> 4a72a78348968a5fc2040d881e72091356356487
