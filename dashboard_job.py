import streamlit as st
import pandas as pd

st.title("🎯 Job Acceptance Prediction Dashboard")

df = pd.read_csv("data/processed/job_acceptance_clean.csv")

st.metric("Total Candidates", len(df))
st.metric("Job Acceptance Rate (%)", round(df["placement_status"].mean()*100, 2))
