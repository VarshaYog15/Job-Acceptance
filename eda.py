"""
EDA MODULE – Job Acceptance Project
----------------------------------
Robust, screen-displayed EDA with safe column checks
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def safe_group_mean(df, group_col, value_col):
    """Safely compute group mean if column exists"""
    if group_col in df.columns:
        return df.groupby(group_col)[value_col].mean()
    else:
        print(f"⚠️ Skipped: '{group_col}' column not found\n")
        return None


def run_eda(df, target_col="status"):

    print("\n==============================")
    print("📊 EDA STARTED")
    print("==============================\n")

    # --------------------------------------------------
    # Placement status → numeric
    # --------------------------------------------------
    df["placement_numeric"] = df[target_col].map({
        "placed": 1,
        "not placed": 0
    })

    print("✅ Placement mapping:")
    print(df["placement_numeric"].value_counts(), "\n")

    # --------------------------------------------------
    # 1️⃣ Academic Scores vs Placement Outcome
    # --------------------------------------------------
    academic_summary = df.groupby(target_col)["academic_avg"].mean()
    print("📘 Academic Avg by Placement:")
    print(academic_summary, "\n")

    academic_summary.plot(kind="bar", title="Academic Avg vs Placement")
    plt.ylabel("Academic Avg")
    plt.show()

    # --------------------------------------------------
    # 2️⃣ Skills Match vs Interview Performance
    # --------------------------------------------------
    corr_val = df["skills_match_percentage"].corr(df["interview_score_avg"])
    print(f"📊 Correlation (Skills vs Interview): {corr_val:.3f}\n")

    plt.scatter(df["skills_match_percentage"], df["interview_score_avg"], alpha=0.4)
    plt.title("Skills Match vs Interview Performance")
    plt.xlabel("Skills Match %")
    plt.ylabel("Interview Avg Score")
    plt.show()

    # --------------------------------------------------
    # 3️⃣ Certification Impact on Job Acceptance
    # --------------------------------------------------
    cert_rate = safe_group_mean(df, "has_certifications", "placement_numeric")
    if cert_rate is not None:
        print("🎓 Certification Impact:")
        print(cert_rate, "\n")
        cert_rate.plot(kind="bar", title="Certification Impact")
        plt.ylabel("Acceptance Rate")
        plt.show()

    # --------------------------------------------------
    # 4️⃣ Acceptance Rate by Company Tier (Optional)
    # --------------------------------------------------
    tier_rate = safe_group_mean(df, "company_tier", "placement_numeric")
    if tier_rate is not None:
        print("🏢 Acceptance by Company Tier:")
        print(tier_rate, "\n")
        tier_rate.plot(kind="bar", title="Acceptance by Company Tier")
        plt.ylabel("Acceptance Rate")
        plt.show()

    # --------------------------------------------------
    # 5️⃣ Experience vs Placement Success
    # --------------------------------------------------
    exp_rate = safe_group_mean(df, "experience_level", "placement_numeric")
    if exp_rate is not None:
        print("💼 Experience vs Placement:")
        print(exp_rate, "\n")
        exp_rate.plot(kind="bar", title="Experience vs Placement")
        plt.ylabel("Acceptance Rate")
        plt.show()

    # --------------------------------------------------
    # 6️⃣ Competition Level Impact
    # --------------------------------------------------
    comp_rate = safe_group_mean(df, "competition_level", "placement_numeric")
    if comp_rate is not None:
        print("⚔️ Competition Level Impact:")
        print(comp_rate, "\n")
        comp_rate.plot(kind="bar", title="Competition Impact")
        plt.ylabel("Acceptance Rate")
        plt.show()

    # --------------------------------------------------
    # 7️⃣ Interview Score vs Placement Probability
    # --------------------------------------------------
    plt.scatter(
        df["interview_score_avg"],
        df["placement_probability_score"],
        alpha=0.4
    )
    plt.title("Interview Score vs Placement Probability")
    plt.xlabel("Interview Score Avg")
    plt.ylabel("Placement Probability")
    plt.show()

    # --------------------------------------------------
    # 8️⃣ Employability Test Score Analysis
    # --------------------------------------------------
    df["employability_score"] = (
        df["technical_score"] + df["aptitude_score"]
    ) / 2

    employ_summary = df.groupby(target_col)["employability_score"].mean()
    print("🧪 Employability Score Analysis:")
    print(employ_summary, "\n")

    employ_summary.plot(kind="bar", title="Employability Score vs Placement")
    plt.ylabel("Employability Score")
    plt.show()

    # --------------------------------------------------
    # 9️⃣ Dropout Risk Identification
    # --------------------------------------------------
    df["dropout_risk"] = (
        (df["placement_probability_score"] < 0.5) &
        (df["long_notice_period"] == 1)
    )

    dropout_pct = df["dropout_risk"].mean() * 100
    print(f"⚠️ Dropout Risk Candidates: {dropout_pct:.2f}%\n")

    # --------------------------------------------------
    # 🔟 Bias-aware Gender Analysis
    # --------------------------------------------------
    gender_rate = safe_group_mean(df, "gender", "placement_numeric")
    if gender_rate is not None:
        print("⚖️ Gender-wise Acceptance:")
        print(gender_rate, "\n")
        gender_rate.plot(kind="bar", title="Gender-wise Acceptance")
        plt.ylabel("Acceptance Rate")
        plt.show()

    print("✅ EDA COMPLETED SUCCESSFULLY\n")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":

    DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/job_acceptance_features.csv"

    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("📊 Running EDA...\n")
    run_eda(df)
