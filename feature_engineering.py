import pandas as pd


def feature_engineering(df, target_col="status"):
    """
    Performs data cleaning and feature engineering.
    Returns a fully feature-engineered DataFrame
    aligned with Step 5 of the project document.
    """

    # -----------------------------
    # 1️⃣ Handle Missing Values
    # -----------------------------
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].str.strip().str.lower()

    # -----------------------------
    # 2️⃣ Interview Score Engineering
    # -----------------------------
    df["interview_score_avg"] = (
        df["technical_score"]
        + df["aptitude_score"]
        + df["communication_score"]
    ) / 3

    # -----------------------------
    # 3️⃣ Experience Category
    # -----------------------------
    df["experience_level"] = pd.cut(
        df["years_of_experience"],
        bins=[-1, 1, 5, 30],
        labels=["fresher", "junior", "senior"]
    )

    # -----------------------------
    # 4️⃣ Skills Match Level
    # -----------------------------
    df["skills_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[0, 40, 70, 100],
        labels=["low", "medium", "high"]
    )

    # -----------------------------
    # 5️⃣ Academic Performance Bands  ✅ (NEW)
    # -----------------------------
    df["academic_avg"] = (
        df["ssc_percentage"]
        + df["hsc_percentage"]
        + df["degree_percentage"]
    ) / 3

    df["academic_performance_band"] = pd.cut(
        df["academic_avg"],
        bins=[0, 60, 75, 100],
        labels=["low", "medium", "high"]
    )

    # -----------------------------
    # 6️⃣ Interview Performance Category ✅ (NEW)
    # -----------------------------
    df["interview_performance_category"] = pd.cut(
        df["interview_score_avg"],
        bins=[0, 50, 75, 100],
        labels=["poor", "average", "excellent"]
    )

    # -----------------------------
    # 7️⃣ Placement Probability Score ✅ (NEW)
    # Rule-based (NOT ML prediction)
    # -----------------------------
    df["placement_probability_score"] = (
        0.4 * df["skills_match_percentage"]
        + 0.3 * df["interview_score_avg"]
        + 0.3 * df["academic_avg"]
    ) / 100

    # -----------------------------
    # 8️⃣ Salary Expectation Gap
    # -----------------------------
    df["ctc_gap"] = df["expected_ctc_lpa"] - df["previous_ctc_lpa"]

    # -----------------------------
    # 9️⃣ Binary Flags
    # -----------------------------
    df["has_certifications"] = (df["certifications_count"] > 0).astype(int)
    df["long_notice_period"] = (df["notice_period_days"] > 60).astype(int)
    df["employment_gap_flag"] = (df["employment_gap_months"] > 0).astype(int)

    # -----------------------------
    # 🔟 One-Hot Encoding
    # -----------------------------
    final_cat_cols = [
        "gender", "degree_specialization", "internship_experience",
        "career_switch_willingness", "relevant_experience", "company_tier",
        "job_role_match", "competition_level", "bond_requirement",
        "layoff_history", "relocation_willingness",
        "experience_level", "skills_level",
        "academic_performance_band", "interview_performance_category"
    ]

    df_encoded = pd.get_dummies(
        df,
        columns=final_cat_cols,
        drop_first=True
    )

    # -----------------------------
    # 1️⃣1️⃣ Ensure Target Column Last
    # -----------------------------
    cols = [c for c in df_encoded.columns if c != target_col] + [target_col]
    df_encoded = df_encoded[cols]

    return df_encoded


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    RAW_DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/cleaned_data.csv"
    OUTPUT_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/job_acceptance_features.csv"

    print("📥 Loading data...")
    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("⚙️ Performing feature engineering...")
    df_features = feature_engineering(df_raw)

    import os

base_path = OUTPUT_PATH
file_path = base_path

# If file is locked, create a new version
counter = 1
while True:
    try:
        df_features.to_csv(file_path, index=False)
        break
    except PermissionError:
        file_path = base_path.replace(".csv", f"_v{counter}.csv")
        counter += 1

print(f"💾 Saved to: {file_path}")
