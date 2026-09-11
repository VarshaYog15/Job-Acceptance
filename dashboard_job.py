
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_data.csv"
FEATURE_DATA_PATH = BASE_DIR / "job_acceptance_features.csv"

sns.set_style("whitegrid")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "placement_numeric" not in df.columns and "status" in df.columns:
        df["placement_numeric"] = df["status"].map({
            "placed": 1,
            "not placed": 0
        })

    if "interview_score_avg" not in df.columns:
        df["interview_score_avg"] = (
            df["technical_score"]
            + df["aptitude_score"]
            + df["communication_score"]
        ) / 3

    return df


@st.cache_data
def load_feature_data():
    return pd.read_csv(FEATURE_DATA_PATH)


# ============================================================
# TRAIN MODEL AUTOMATICALLY
# No .pkl files are required.
# ============================================================
@st.cache_resource
def train_prediction_model():
    fdf = load_feature_data()

    # Target
    y = fdf["status"].map({
        "placed": 1,
        "not placed": 0
    }).astype(int)

    # Keep exactly the 45 features used by the supplied
    # feature-engineered dataset.
    feature_names = [c for c in fdf.columns if c != "status"]

    X = fdf[feature_names].copy()

    # Convert booleans to numeric
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    X = X.astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test_scaled)
    )

    # placement_probability_score is one of the 45 features.
    # Train a helper regressor so a new candidate can receive
    # this feature before classification.
    score_features = [
        c for c in feature_names
        if c != "placement_probability_score"
    ]

    score_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )

    score_model.fit(
        X[score_features],
        fdf["placement_probability_score"].astype(float)
    )

    return model, scaler, feature_names, score_model, score_features, accuracy


df = load_data()
model, scaler, feature_names, score_model, score_features, model_accuracy = train_prediction_model()

# ============================================================
# TITLE
# ============================================================
st.title("📊 Job Acceptance Analytics Dashboard")
st.caption("Streamlit + Random Forest Job Acceptance Prediction")

# ============================================================
# KPI CALCULATIONS
# ============================================================
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

# ============================================================
# KPI DISPLAY
# ============================================================
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

# ============================================================
# VISUALIZATIONS
# ============================================================
st.subheader("📈 Insights")

st.write("### Placement Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="status", ax=ax1)
ax1.set_title("Placement Distribution")
st.pyplot(fig1)
plt.close(fig1)

st.write("### Skills vs Placement")

df["skills_level"] = pd.cut(
    df["skills_match_percentage"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

skills_summary = (
    df.groupby("skills_level", observed=False)["placement_numeric"]
    .mean() * 100
)

fig2, ax2 = plt.subplots()
skills_summary.plot(kind="bar", ax=ax2)
ax2.set_title("Placement Rate by Skill Level")
ax2.set_ylabel("Acceptance Rate (%)")
ax2.set_ylim(0, 100)
st.pyplot(fig2)
plt.close(fig2)

st.write("### Interview Score vs Placement")

df["interview_level"] = pd.cut(
    df["interview_score_avg"],
    bins=[0, 40, 60, 80, 100],
    labels=["Low", "Average", "Good", "Excellent"],
    include_lowest=True
)

interview_summary = (
    df.groupby("interview_level", observed=False)["placement_numeric"]
    .mean() * 100
)

fig3, ax3 = plt.subplots()
interview_summary.plot(kind="bar", ax=ax3)
ax3.set_title("Placement Rate by Interview Level")
ax3.set_ylabel("Acceptance Rate (%)")
ax3.set_ylim(0, 100)
st.pyplot(fig3)
plt.close(fig3)

st.write("### High-Risk Candidates")

fig4, ax4 = plt.subplots()
sns.histplot(df["skills_match_percentage"], bins=20, ax=ax4)
ax4.set_title("Skills Distribution")
ax4.set_xlabel("Skills Match Percentage")
st.pyplot(fig4)
plt.close(fig4)

# ============================================================
# PREDICTION
# ============================================================
st.markdown("---")
st.header("🤖 Predict Job Acceptance")

st.write(
    "Enter the candidate details and click the button below. "
    "The dashboard automatically trains the Random Forest model "
    "from the supplied feature-engineered dataset, so no .pkl "
    "model files are required."
)

st.info(f"Model test accuracy: {model_accuracy * 100:.2f}%")

st.subheader("👤 Candidate Information")

c1, c2, c3 = st.columns(3)

with c1:
    age_years = st.number_input("Age (years)", 18, 60, 25)
    gender = st.selectbox("Gender", ["female", "male"])
    ssc_percentage = st.number_input("SSC Percentage", 0.0, 100.0, 70.0, step=0.1)
    hsc_percentage = st.number_input("HSC Percentage", 0.0, 100.0, 70.0, step=0.1)
    degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 70.0, step=0.1)
    degree_specialization = st.selectbox(
        "Degree Specialization",
        ["computer science", "electronics", "information technology",
         "mechanical", "others"]
    )
    technical_score = st.number_input("Technical Score", 0.0, 100.0, 70.0, step=0.1)

with c2:
    aptitude_score = st.number_input("Aptitude Score", 0.0, 100.0, 70.0, step=0.1)
    communication_score = st.number_input("Communication Score", 0.0, 100.0, 70.0, step=0.1)
    skills_match_percentage = st.number_input(
        "Skills Match Percentage", 0.0, 100.0, 70.0, step=0.1
    )
    certifications_count = st.number_input("Certifications Count", 0, 20, 1)
    internship_experience = st.selectbox("Internship Experience", ["no", "yes"])
    years_of_experience = st.number_input("Years of Experience", 0, 30, 1)
    career_switch_willingness = st.selectbox(
        "Career Switch Willingness", ["not willing", "willing"]
    )
    relevant_experience = st.selectbox(
        "Relevant Experience", ["not relevant", "relevant"]
    )

with c3:
    previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 0.0, 100.0, 5.0, step=0.1)
    expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 0.0, 100.0, 6.0, step=0.1)
    company_tier = st.selectbox("Company Tier", ["tier 1", "tier 2", "tier 3"])
    job_role_match = st.selectbox("Job Role Match", ["matched", "not matched"])
    competition_level = st.selectbox(
        "Competition Level", ["high", "medium", "low"]
    )
    bond_requirement = st.selectbox(
        "Bond Requirement", ["not required", "required"]
    )
    notice_period_days = st.number_input(
        "Notice Period (days)", 0.0, 180.0, 30.0, step=1.0
    )
    layoff_history = st.selectbox("Layoff History", ["no", "yes"])
    employment_gap_months = st.number_input(
        "Employment Gap (months)", 0.0, 120.0, 0.0, step=1.0
    )
    relocation_willingness = st.selectbox(
        "Relocation Willingness", ["not willing", "willing"]
    )

interview_score_avg = (
    technical_score + aptitude_score + communication_score
) / 3

academic_avg = (
    ssc_percentage + hsc_percentage + degree_percentage
) / 3

ctc_gap = expected_ctc_lpa - previous_ctc_lpa

st.subheader("📊 Calculated Candidate Metrics")

m1, m2, m3 = st.columns(3)
m1.metric("Interview Average", f"{interview_score_avg:.2f}")
m2.metric("Academic Average", f"{academic_avg:.2f}")
m3.metric("CTC Gap", f"{ctc_gap:.2f} LPA")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_raw_features():
    row = {}

    # Numeric features
    row["age_years"] = age_years
    row["ssc_percentage"] = ssc_percentage
    row["hsc_percentage"] = hsc_percentage
    row["degree_percentage"] = degree_percentage
    row["technical_score"] = technical_score
    row["aptitude_score"] = aptitude_score
    row["communication_score"] = communication_score
    row["skills_match_percentage"] = skills_match_percentage
    row["certifications_count"] = certifications_count
    row["years_of_experience"] = years_of_experience
    row["previous_ctc_lpa"] = previous_ctc_lpa
    row["expected_ctc_lpa"] = expected_ctc_lpa
    row["notice_period_days"] = notice_period_days
    row["employment_gap_months"] = employment_gap_months
    row["interview_score_avg"] = interview_score_avg
    row["academic_avg"] = academic_avg
    row["ctc_gap"] = ctc_gap

    # Binary features
    row["has_certifications"] = int(certifications_count > 0)
    row["long_notice_period"] = int(notice_period_days > 30)
    row["employment_gap_flag"] = int(employment_gap_months > 0)

    # One-hot categorical features
    row["gender_male"] = int(gender == "male")

    row["degree_specialization_electronics"] = int(
        degree_specialization == "electronics"
    )
    row["degree_specialization_information technology"] = int(
        degree_specialization == "information technology"
    )
    row["degree_specialization_mechanical"] = int(
        degree_specialization == "mechanical"
    )
    row["degree_specialization_others"] = int(
        degree_specialization == "others"
    )

    row["internship_experience_yes"] = int(internship_experience == "yes")
    row["career_switch_willingness_willing"] = int(
        career_switch_willingness == "willing"
    )
    row["relevant_experience_relevant"] = int(
        relevant_experience == "relevant"
    )

    row["company_tier_tier 2"] = int(company_tier == "tier 2")
    row["company_tier_tier 3"] = int(company_tier == "tier 3")

    row["job_role_match_not matched"] = int(
        job_role_match == "not matched"
    )

    row["competition_level_low"] = int(competition_level == "low")
    row["competition_level_medium"] = int(competition_level == "medium")

    row["bond_requirement_required"] = int(
        bond_requirement == "required"
    )
    row["layoff_history_yes"] = int(layoff_history == "yes")
    row["relocation_willingness_willing"] = int(
        relocation_willingness == "willing"
    )

    # Engineered categories
    row["experience_level_junior"] = int(years_of_experience >= 2)
    row["experience_level_senior"] = 0

    row["skills_level_medium"] = int(
        40 < skills_match_percentage <= 70
    )
    row["skills_level_high"] = int(
        skills_match_percentage > 70
    )

    row["academic_performance_band_medium"] = int(
        60 < academic_avg <= 75
    )
    row["academic_performance_band_high"] = int(
        academic_avg > 75
    )

    row["interview_performance_category_average"] = int(
        50 < interview_score_avg <= 75
    )
    row["interview_performance_category_excellent"] = int(
        interview_score_avg > 75
    )

    return pd.DataFrame([row])


# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("🔮 Predict Job Acceptance", type="primary"):

    X_new = make_raw_features()

    # Add any missing feature columns as zero
    for col in score_features:
        if col not in X_new.columns:
            X_new[col] = 0

    # Exact order expected by the placement score model
    X_score = X_new[score_features].astype(float)

    # Predict the engineered placement probability score
    placement_score = float(score_model.predict(X_score)[0])
    placement_score = float(np.clip(placement_score, 0, 1))

    # Add the final engineered feature
    X_new["placement_probability_score"] = placement_score

    # Add any remaining feature columns and preserve exact order
    for col in feature_names:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[feature_names].astype(float)

    # Scale exactly as during model training
    X_scaled = scaler.transform(X_new)

    # Final prediction
    prediction = int(model.predict(X_scaled)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[0]
        rejection_probability = float(probabilities[0] * 100)
        acceptance_probability = float(probabilities[1] * 100)
    else:
        acceptance_probability = 100.0 if prediction == 1 else 0.0
        rejection_probability = 100.0 - acceptance_probability

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("✅ JOB ACCEPTED / PLACED")
    else:
        st.error("❌ JOB NOT ACCEPTED / NOT PLACED")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Acceptance Probability",
        f"{acceptance_probability:.2f}%"
    )
    r2.metric(
        "Rejection Probability",
        f"{rejection_probability:.2f}%"
    )
    r3.metric(
        "Placement Score",
        f"{placement_score * 100:.2f}%"
    )

    st.progress(
        int(round(acceptance_probability)),
        text=f"Acceptance probability: {acceptance_probability:.2f}%"
    )

    if acceptance_probability >= 75:
        st.info("🟢 High likelihood of job acceptance")
    elif acceptance_probability >= 50:
        st.info("🟡 Moderate likelihood of job acceptance")
    else:
        st.info("🔴 Low likelihood of job acceptance")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "📊 Job Acceptance Dashboard | Streamlit + Random Forest | "
    "No external .pkl files required"
)
