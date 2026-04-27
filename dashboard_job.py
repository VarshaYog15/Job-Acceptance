<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 📁 CONFIG
# -------------------------------
DATA_PATH = "cleaned_data.csv"

sns.set_style("whitegrid")

# -------------------------------
# 📥 LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Target column
    if "placement_numeric" not in df.columns and "status" in df.columns:
        df["placement_numeric"] = df["status"].map({
            "placed": 1,
            "not placed": 0
        })

    # Interview average
    if "interview_score_avg" not in df.columns:
        df["interview_score_avg"] = (
            df["technical_score"] +
            df["aptitude_score"] +
            df["communication_score"]
        ) / 3

    return df

df = load_data()

# -------------------------------
# 🎯 TITLE
# -------------------------------
st.title("📊 Job Acceptance Analytics Dashboard")

# -------------------------------
# 📊 KPI CALCULATIONS
# -------------------------------
total_candidates = len(df)
placement_rate = df["placement_numeric"].mean() * 100
avg_interview = df["interview_score_avg"].mean()
avg_skills = df["skills_match_percentage"].mean()

offer_dropout = 100 - placement_rate

high_risk = df[
    (df["skills_match_percentage"] < 50) |
    (df["interview_score_avg"] < 60)
]
high_risk_pct = (len(high_risk) / total_candidates) * 100

# -------------------------------
# 🧾 KPI DISPLAY
# -------------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8 = st.columns(2)

col1.metric("Total Candidates", f"{total_candidates:,}")
col2.metric("Placement Rate", f"{placement_rate:.2f}%")
col3.metric("Job Acceptance Rate", f"{placement_rate:.2f}%")

col4.metric("Avg Interview Score", f"{avg_interview:.2f}")
col5.metric("Avg Skills Match", f"{avg_skills:.2f}%")
col6.metric("Offer Dropout Rate", f"{offer_dropout:.2f}%")

col7.metric("High-Risk Candidates", f"{high_risk_pct:.2f}%")
col8.metric("High-Risk Count", f"{len(high_risk)}")

# -------------------------------
# 📊 VISUALIZATIONS (NO PYARROW)
# -------------------------------
st.subheader("📈 Insights")

# 1️⃣ Placement Distribution
st.write("### Placement Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="status", ax=ax1)
ax1.set_title("Placement Distribution")
st.pyplot(fig1)

# 2️⃣ Skills vs Placement
st.write("### Skills vs Placement")

df["skills_level"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

skills_summary = df.groupby("skills_level")["placement_numeric"].mean()

fig2, ax2 = plt.subplots()
skills_summary.plot(kind="bar", ax=ax2)
ax2.set_title("Placement Rate by Skill Level")
ax2.set_ylabel("Acceptance Rate")
st.pyplot(fig2)

# 3️⃣ Interview vs Placement
st.write("### Interview Score vs Placement")

df["interview_level"] = pd.cut(
    df["interview_score_avg"],
    bins=[0, 40, 60, 80, 100],
    labels=["Low", "Average", "Good", "Excellent"],
    include_lowest=True
)

interview_summary = df.groupby("interview_level")["placement_numeric"].mean()

fig3, ax3 = plt.subplots()
interview_summary.plot(kind="bar", ax=ax3)
ax3.set_title("Placement Rate by Interview Level")
ax3.set_ylabel("Acceptance Rate")
st.pyplot(fig3)

# 4️⃣ High-Risk Distribution
st.write("### High-Risk Candidates")

fig4, ax4 = plt.subplots()
sns.histplot(df["skills_match_percentage"], bins=20, ax=ax4)
ax4.set_title("Skills Distribution")
st.pyplot(fig4)

# -------------------------------
# 📌 FOOTER
# -------------------------------
st.markdown("---")
st.caption("📊 Job Acceptance Dashboard | Streamlit + Matplotlib (No pyarrow)")
=======
import streamlit as st
import pandas as pd

st.title("🎯 Job Acceptance Prediction Dashboard")

df = pd.read_csv("data/processed/job_acceptance_clean.csv")

st.metric("Total Candidates", len(df))
st.metric("Job Acceptance Rate (%)", round(df["placement_status"].mean()*100, 2))
>>>>>>> 4a72a78348968a5fc2040d881e72091356356487
