import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/cleaned_data.csv"
OUTPUT_DIR = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")


# -----------------------------
# FEATURE CREATION
# -----------------------------
def prepare_data(df):

    # Target
    df["placement_numeric"] = df["status"].map({
        "placed": 1,
        "not placed": 0
    })

    # Interview score
    df["interview_score_avg"] = (
        df["technical_score"] +
        df["aptitude_score"] +
        df["communication_score"]
    ) / 3

    # Academic avg + band
    df["academic_avg"] = df[
        ["ssc_percentage", "hsc_percentage", "degree_percentage"]
    ].mean(axis=1)

    df["academic_band"] = pd.cut(
        df["academic_avg"],
        bins=[0, 50, 70, 85, 100],
        labels=["Poor", "Average", "Good", "Excellent"]
    )

    # Skills level
    df["skills_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"]
    )

    # Experience
    df["experience_level"] = pd.cut(
        df["years_of_experience"],
        bins=[-1, 1, 5, 50],
        labels=["Fresher", "Junior", "Senior"]
    )

    # Certifications
    if "certifications_count" in df.columns:
        df["has_certifications"] = (df["certifications_count"] > 0).astype(int)

    # Interview category
    df["interview_category"] = pd.cut(
        df["interview_score_avg"],
        bins=[0, 40, 60, 80, 100],
        labels=["Low", "Average", "Good", "Excellent"]
    )

    return df


# -----------------------------
# ANALYSIS FUNCTIONS
# -----------------------------

# 1️⃣ Academic vs Placement

# 📁 Paths
DATA_PATH = "cleaned_data.csv"   # change if needed
OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def academic_analysis(df):
    print("\n📘 Academic Performance vs Placement")

    # -------------------------------
    # 0️⃣ Safety checks
    # -------------------------------
    if "academic_avg" not in df.columns:
        print("⚠️ 'academic_avg' column not found")
        return

    if "placement_numeric" not in df.columns:
        print("⚠️ 'placement_numeric' column not found")
        return

    # -------------------------------
    # 1️⃣ Create Academic Bands
    # -------------------------------
    df["academic_band"] = pd.cut(
        df["academic_avg"],
        bins=[0, 60, 75, 85, 100],
        labels=["Poor", "Average", "Good", "Excellent"],
        include_lowest=True
    )

    # Remove empty categories (IMPORTANT FIX)
    df = df.dropna(subset=["academic_band"])

    # -------------------------------
    # 2️⃣ Compute Summary
    # -------------------------------
    summary = (
        df.groupby("academic_band", observed=True)["placement_numeric"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary.columns = ["academic_band", "acceptance_rate", "candidate_count"]

    print("\n📊 Summary:")
    print(summary)

    # -------------------------------
    # 3️⃣ Plot: Acceptance Rate + Count (COMBINED)
    # -------------------------------
    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=summary,
        x="academic_band",
        y="acceptance_rate"
    )

    # Add labels with count
    for i, row in summary.iterrows():
        ax.text(
            i,
            row["acceptance_rate"] + 0.02,
            f"{row['acceptance_rate']:.2f}\n(n={int(row['candidate_count'])})",
            ha="center",
            fontsize=10
        )

    plt.title("Academic Performance vs Job Acceptance Rate")
    plt.xlabel("Academic Band")
    plt.ylabel("Acceptance Rate")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/academic_analysis.png")
    plt.close()

    print("✅ Academic analysis saved → eda_outputs/academic_analysis.png")


def main():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Ensure target is numeric
    if "placement_numeric" not in df.columns:
        if "status" in df.columns:
            df["placement_numeric"] = df["status"].map({
                "placed": 1,
                "not placed": 0
            })
        else:
            print("⚠️ No target column found")
            return

    academic_analysis(df)


if __name__ == "__main__":
    main()


# 2️⃣ Skills vs Interview
def skills_analysis(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -----------------------------
    # Ensure clean labels
    # -----------------------------
    df["skills_level"] = df["skills_level"].astype(str)

    # -----------------------------
    # 1️⃣ Placement Rate by Skill Level
    # -----------------------------
    placement = (
        df.groupby("skills_level")["placement_numeric"]
        .mean()
        .reset_index()
    )

    print("\n📊 Placement Rate by Skill Level:")
    print(placement)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=placement, x="skills_level", y="placement_numeric")

    plt.title("Placement Rate by Skill Level")
    plt.xlabel("Skill Level")
    plt.ylabel("Placement Rate")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/skills_vs_placement.png")
    plt.show()

    # -----------------------------
    # 2️⃣ Interview Score by Skill Level
    # -----------------------------
    interview = (
        df.groupby("skills_level")["interview_score_avg"]
        .mean()
        .reset_index()
    )

    print("\n📊 Interview Performance by Skill Level:")
    print(interview)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=interview, x="skills_level", y="interview_score_avg")

    plt.title("Interview Score by Skill Level")
    plt.xlabel("Skill Level")
    plt.ylabel("Average Interview Score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/skills_vs_interview.png")
    plt.show()

# 3️⃣ Certification Impact
# 📁 Output folder
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def certification_analysis(df):
    print("\n🎓 Certification Impact Analysis")

    # -------------------------------
    # 0️⃣ Safety check
    # -------------------------------
    if "has_certifications" not in df.columns:
        print("⚠️ 'has_certifications' column not found")
        return

    if "placement_numeric" not in df.columns:
        print("⚠️ 'placement_numeric' column not found")
        return

    # -------------------------------
    # 1️⃣ Clean & Map Labels
    # -------------------------------
    df["cert_label"] = df["has_certifications"].map({
        0: "No Certification",
        1: "Has Certification"
    })

    # -------------------------------
    # 2️⃣ Acceptance Rate (MEAN)
    # -------------------------------
    summary = (
        df.groupby("cert_label", observed=True)["placement_numeric"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary.rename(columns={
        "mean": "acceptance_rate",
        "count": "candidate_count"
    }, inplace=True)

    print("\n📊 Summary:")
    print(summary)

    # -------------------------------
    # 3️⃣ Plot 1: Acceptance Rate
    # -------------------------------
    plt.figure(figsize=(6, 4))

    ax = sns.barplot(
        data=summary,
        x="cert_label",
        y="acceptance_rate"
    )

    # Add values on top
    for i, row in summary.iterrows():
        ax.text(i, row["acceptance_rate"] + 0.01,
                f"{row['acceptance_rate']:.2f}",
                ha='center', fontsize=10)

    plt.title("Certification Impact on Job Acceptance")
    plt.xlabel("Certification Status")
    plt.ylabel("Acceptance Rate")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/certification_impact.png")
    plt.close()

    # -------------------------------
    # 4️⃣ Plot 2: Count Distribution (IMPORTANT)
    # -------------------------------
    plt.figure(figsize=(6, 4))

    ax = sns.countplot(
        data=df,
        x="cert_label",
        hue="status"
    )

    # Add count labels
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 200,
                f"{height}",
                ha="center",
                fontsize=9
            )

    plt.title("Certification vs Placement Count")
    plt.xlabel("Certification Status")
    plt.ylabel("Number of Candidates")
    plt.legend(title="Placement Status")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/certification_count.png")
    plt.close()

    print("✅ Certification analysis saved in eda_outputs/")


# 4️⃣ Company Tier
def company_tier_analysis(df):
    if "company_tier" in df.columns:
        plt.figure()
        sns.barplot(
            data=df,
            x="company_tier",
            y="placement_numeric"
        )
        plt.title("Acceptance Rate by Company Tier")
        plt.savefig(f"{OUTPUT_DIR}/company_tier.png")
        plt.close()


# 5️⃣ Experience vs Placement
def experience_analysis(df):
    plt.figure()
    sns.barplot(
        data=df,
        x="experience_level",
        y="placement_numeric"
    )
    plt.title("Experience vs Placement Success")
    plt.savefig(f"{OUTPUT_DIR}/experience.png")
    plt.close()


# 6️⃣ Competition Level
def competition_analysis(df):
    if "competition_level" in df.columns:
        plt.figure()
        sns.barplot(
            data=df,
            x="competition_level",
            y="placement_numeric"
        )
        plt.title("Competition Level Impact on Acceptance")
        plt.savefig(f"{OUTPUT_DIR}/competition.png")
        plt.close()


# 7️⃣ Interview vs Placement Probability
def interview_analysis(df):
    plt.figure()
    sns.barplot(
        data=df,
        x="interview_category",
        y="placement_numeric"
    )
    plt.title("Interview Score vs Placement Probability")
    plt.savefig(f"{OUTPUT_DIR}/interview_vs_probability.png")
    plt.close()


# 8️⃣ Employability Test Analysis
def employability_analysis(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------------------------------------------------
    # 1️⃣ Create Score Bands
    # --------------------------------------------------
    df["aptitude_band"] = pd.cut(
        df["aptitude_score"],
        bins=[0, 50, 60, 70, 80, 90, 100],
        labels=["<50", "50-60", "60-70", "70-80", "80-90", "90+"]
    )

    # --------------------------------------------------
    # 2️⃣ Calculate Placement Rate
    # --------------------------------------------------
    analysis = (
        df.groupby("aptitude_band")["placement_numeric"]
        .mean()
        .reset_index()
    )

    print("\n📊 Employability Analysis (Aptitude vs Placement):")
    print(analysis)

    # --------------------------------------------------
    # 3️⃣ Plot (Clear Insight)
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(data=analysis, x="aptitude_band", y="placement_numeric")

    plt.title("Placement Rate by Aptitude Score Band")
    plt.xlabel("Aptitude Score Band")
    plt.ylabel("Placement Rate")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/employability.png")
    plt.show()


# 9️⃣ Dropout Risk
def dropout_analysis(df):
    if "employment_gap_months" in df.columns:
        plt.figure()
        sns.boxplot(
            data=df,
            x="status",
            y="employment_gap_months"
        )
        plt.title("Dropout Risk (Employment Gap Analysis)")
        plt.savefig(f"{OUTPUT_DIR}/dropout_risk.png")
        plt.close()


# 🔟 Feature Importance
def feature_importance(df):
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Importance (Correlation Heatmap)")
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    plt.close()


def bias_analysis(df):
    if "gender" in df.columns:

        # Prepare data
        summary = df.groupby("gender", observed=True).agg(
            acceptance_rate=("placement_numeric", "mean"),
            total_candidates=("placement_numeric", "count")
        ).reset_index()

        # Improve readability
        plt.figure(figsize=(8, 5))

        # Barplot
        ax = sns.barplot(
            data=summary,
            x="gender",
            y="acceptance_rate"
        )

        plt.title("Gender Bias Analysis", fontsize=14)
        plt.ylabel("Acceptance Rate")
        plt.xlabel("Gender")

        # Add acceptance labels (TOP of bars)
        for i, val in enumerate(summary["acceptance_rate"]):
            ax.text(i, val + 0.01, f"{val:.2f}", ha='center', fontsize=11)

        # Add candidate count BELOW bars
        for i, val in enumerate(summary["total_candidates"]):
            ax.text(i, 0.02, f"n={val}", ha='center', fontsize=10, color="black")

        plt.ylim(0, max(summary["acceptance_rate"]) + 0.1)

        plt.savefig(f"{OUTPUT_DIR}/bias_gender.png")
        plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    df = prepare_data(df)

    print("📊 Running Full Analyst EDA...")

    academic_analysis(df)
    skills_analysis(df)
    certification_analysis(df)
    company_tier_analysis(df)
    experience_analysis(df)
    competition_analysis(df)
    interview_analysis(df)
    employability_analysis(df)
    dropout_analysis(df)
    feature_importance(df)
    bias_analysis(df)

    print("✅ ALL ANALYSIS COMPLETED → Check eda_outputs folder")