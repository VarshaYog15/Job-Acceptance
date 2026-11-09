import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    FEATURE_DATA_PATH,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    PLOTS_DIR
)

# ============================================================
# CONFIGURATION
# ============================================================

sns.set_theme(style="whitegrid")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n📥 Loading feature-engineered data...")

    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"\nFeature-engineered dataset not found:\n"
            f"{FEATURE_DATA_PATH}\n\n"
            "Please run feature_engineering.py first."
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    print("✅ Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    return df


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(df):

    df = df.copy()

    df["placement_numeric"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "placed": 1,
            "not placed": 0,
            "not_placed": 0
        })
    )

    return df


# ============================================================
# 1. ACADEMIC SCORES VS PLACEMENT OUTCOME
# ============================================================

def academic_scores_vs_placement(df):

    print("\n1️⃣ Academic Scores vs Placement Outcome")

    academic_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]

    available_columns = [
        col for col in academic_columns
        if col in df.columns
    ]

    if not available_columns:
        print("⚠️ Academic columns not found.")
        return

    plot_data = (
        df.groupby("status")[available_columns]
        .mean()
        .reset_index()
        .melt(
            id_vars="status",
            var_name="academic_score",
            value_name="average_score"
        )
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_data,
        x="academic_score",
        y="average_score",
        hue="status"
    )

    plt.title(
        "Academic Scores vs Placement Outcome"
    )

    plt.xlabel("Academic Score")
    plt.ylabel("Average Score (%)")

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "01_academic_scores_vs_placement.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 2. SKILLS MATCH VS INTERVIEW PERFORMANCE
# ============================================================

def skills_match_vs_interview(df):

    print("\n2️⃣ Skills Match vs Interview Performance")

    required_columns = [
        "skills_match_percentage",
        "interview_score_avg"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return

    df_plot = df[
        required_columns + ["status"]
    ].dropna().copy()

    # --------------------------------------------------------
    # Create meaningful ranges
    # --------------------------------------------------------

    df_plot["skills_range"] = pd.cut(
        df_plot["skills_match_percentage"],
        bins=[
            40, 50, 60, 70, 80, 90, 100
        ],
        labels=[
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%"
        ],
        include_lowest=True
    )

    df_plot["interview_range"] = pd.cut(
        df_plot["interview_score_avg"],
        bins=[
            40, 50, 60, 70, 80, 90, 100
        ],
        labels=[
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100"
        ],
        include_lowest=True
    )

    # --------------------------------------------------------
    # Calculate placement rate for each combination
    # --------------------------------------------------------

    df_plot["placed_numeric"] = (
        df_plot["status"]
        .astype(str)
        .str.lower()
        .map({
            "placed": 1,
            "not placed": 0
        })
    )

    heatmap_data = (
        df_plot
        .pivot_table(
            index="interview_range",
            columns="skills_range",
            values="placed_numeric",
            aggfunc="mean",
            observed=False
        )
        * 100
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 7)
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0,
        vmax=100,
        cbar_kws={
            "label": "Placement Rate (%)"
        }
    )

    plt.title(
        "Skills Match vs Interview Performance\n"
        "Placement Rate by Candidate Segment",
        fontsize=16,
        pad=15
    )

    plt.xlabel(
        "Skills Match Percentage"
    )

    plt.ylabel(
        "Average Interview Score"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "02_skills_match_vs_interview.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✅ Skills Match vs Interview heatmap saved."
    )

# ============================================================
# 3. CERTIFICATION IMPACT ON JOB ACCEPTANCE
# ============================================================

def certification_impact(df):

    print("\n3️⃣ Certification Impact on Job Acceptance")

    if "certifications_count" not in df.columns:
        print("⚠️ certifications_count not found.")
        return

    df = df.copy()

    df["certification_status"] = np.where(
        df["certifications_count"] > 0,
        "Certified",
        "No Certification"
    )

    summary = (
        df.groupby("certification_status")
        ["placement_numeric"]
        .mean()
        .mul(100)
        .reset_index(
            name="acceptance_rate"
        )
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=summary,
        x="certification_status",
        y="acceptance_rate"
    )

    plt.title(
        "Certification Impact on Job Acceptance"
    )

    plt.xlabel("Certification Status")
    plt.ylabel("Acceptance Rate (%)")

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "03_certification_impact.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 4. ACCEPTANCE RATE BY COMPANY TIER
# ============================================================

def company_tier_acceptance(df):

    print("\n4️⃣ Acceptance Rate by Company Tier")

    if "company_tier" not in df.columns:
        print("⚠️ company_tier not found.")
        return

    summary = (
        df.groupby("company_tier")
        ["placement_numeric"]
        .mean()
        .mul(100)
        .reset_index(
            name="acceptance_rate"
        )
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=summary,
        x="company_tier",
        y="acceptance_rate"
    )

    plt.title(
        "Acceptance Rate by Company Tier"
    )

    plt.xlabel("Company Tier")
    plt.ylabel("Acceptance Rate (%)")

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "04_company_tier_acceptance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. EXPERIENCE VS PLACEMENT SUCCESS
# ============================================================

def experience_vs_placement(df):

    print("\n5️⃣ Experience vs Placement Success")

    if "years_of_experience" not in df.columns:
        print("⚠️ years_of_experience not found.")
        return

    # Use experience_level if already created
    if "experience_level" in df.columns:

        summary = (
            df.groupby(
                "experience_level",
                observed=True
            )["placement_numeric"]
            .mean()
            .mul(100)
            .reset_index(
                name="placement_rate"
            )
        )

        plt.figure(figsize=(9, 6))

        sns.barplot(
            data=summary,
            x="experience_level",
            y="placement_rate"
        )

        plt.title(
            "Experience Level vs Placement Success"
        )

        plt.xlabel("Experience Level")
        plt.ylabel("Placement Rate (%)")

        plt.ylim(0, 100)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "05_experience_vs_placement.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    else:

        plt.figure(figsize=(9, 6))

        sns.boxplot(
            data=df,
            x="status",
            y="years_of_experience"
        )

        plt.title(
            "Years of Experience vs Placement Outcome"
        )

        plt.xlabel("Placement Status")
        plt.ylabel("Years of Experience")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "05_experience_vs_placement.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 6. COMPETITION LEVEL IMPACT ON ACCEPTANCE
# ============================================================

def competition_impact(df):

    print("\n6️⃣ Competition Level Impact on Acceptance")

    if "competition_level" not in df.columns:
        print("⚠️ competition_level not found.")
        return

    summary = (
        df.groupby("competition_level")
        ["placement_numeric"]
        .mean()
        .mul(100)
        .reset_index(
            name="acceptance_rate"
        )
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=summary,
        x="competition_level",
        y="acceptance_rate"
    )

    plt.title(
        "Competition Level Impact on Acceptance"
    )

    plt.xlabel("Competition Level")
    plt.ylabel("Acceptance Rate (%)")

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "06_competition_level_acceptance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. INTERVIEW SCORE VS PLACEMENT PROBABILITY
# ============================================================

def interview_vs_probability(df):

    print("\n7️⃣ Interview Score vs Placement Probability")

    required_columns = [
        "interview_score_avg",
        "placement_probability_score"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return

    df_plot = df[
        required_columns + ["status"]
    ].dropna().copy()

    # --------------------------------------------------------
    # Convert placement probability score to percentage
    # --------------------------------------------------------

    df_plot["placement_probability_percent"] = (
        df_plot["placement_probability_score"] * 100
    )

    # --------------------------------------------------------
    # Create interview score ranges
    # --------------------------------------------------------

    df_plot["interview_range"] = pd.cut(
        df_plot["interview_score_avg"],
        bins=[
            40, 50, 55, 60, 65, 70,
            75, 80, 85, 90, 100
        ],
        labels=[
            "40-50",
            "50-55",
            "55-60",
            "60-65",
            "65-70",
            "70-75",
            "75-80",
            "80-85",
            "85-90",
            "90-100"
        ],
        include_lowest=True
    )

    # --------------------------------------------------------
    # Calculate average probability for each range
    # --------------------------------------------------------

    summary = (
        df_plot
        .groupby(
            ["interview_range", "status"],
            observed=False
        )[
            "placement_probability_percent"
        ]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # LINE PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 7)
    )

    sns.lineplot(
        data=summary,
        x="interview_range",
        y="placement_probability_percent",
        hue="status",
        marker="o",
        linewidth=3,
        markersize=8
    )

    plt.title(
        "Interview Score vs Placement Probability",
        fontsize=16,
        pad=15
    )

    plt.xlabel(
        "Average Interview Score Range"
    )

    plt.ylabel(
        "Average Placement Probability Score (%)"
    )

    plt.ylim(0, 100)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend(
        title="Placement Status"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "07_interview_vs_placement_probability.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✅ Interview Score vs Placement Probability "
        "plot saved."
    )

# ============================================================
# 8. EMPLOYABILITY TEST SCORE ANALYSIS
# ============================================================

def employability_test_analysis(df):

    print("\n8️⃣ Employability Test Score Analysis")

    test_columns = [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    available_columns = [
        col
        for col in test_columns
        if col in df.columns
    ]

    if not available_columns:
        print("⚠️ Employability test columns not found.")
        return

    plot_data = (
        df.groupby("status")[available_columns]
        .mean()
        .reset_index()
        .melt(
            id_vars="status",
            var_name="test_component",
            value_name="average_score"
        )
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_data,
        x="test_component",
        y="average_score",
        hue="status"
    )

    plt.title(
        "Employability Test Scores vs Placement"
    )

    plt.xlabel(
        "Employability Test Component"
    )

    plt.ylabel(
        "Average Score"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "08_employability_test_scores.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. DROPOUT RISK IDENTIFICATION
# ============================================================

def dropout_risk_identification(df):

    print("\n9️⃣ Dropout Risk Identification")

    df = df.copy()

    risk_conditions = []

    # Low skills match
    if "skills_match_percentage" in df.columns:

        risk_conditions.append(
            df["skills_match_percentage"] < 50
        )

    # Low interview performance
    if "interview_score_avg" in df.columns:

        risk_conditions.append(
            df["interview_score_avg"] < 60
        )

    # Employment gap
    if "employment_gap_months" in df.columns:

        risk_conditions.append(
            df["employment_gap_months"] > 12
        )

    # Long notice period
    if "notice_period_days" in df.columns:

        risk_conditions.append(
            df["notice_period_days"] > 60
        )

    if not risk_conditions:

        print(
            "⚠️ No risk indicators available."
        )

        return

    df["risk_score"] = sum(
        condition.astype(int)
        for condition in risk_conditions
    )

    df["dropout_risk"] = pd.cut(
        df["risk_score"],
        bins=[-1, 0, 1, len(risk_conditions)],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        include_lowest=True
    )

    summary = (
        df.groupby(
            "dropout_risk",
            observed=True
        )["placement_numeric"]
        .mean()
        .mul(100)
        .reset_index(
            name="placement_rate"
        )
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=summary,
        x="dropout_risk",
        y="placement_rate"
    )

    plt.title(
        "Placement Rate by Dropout Risk Level"
    )

    plt.xlabel(
        "Dropout Risk Level"
    )

    plt.ylabel(
        "Placement Rate (%)"
    )

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "09_dropout_risk.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# ============================================================

def feature_importance_analysis():

    print("\n🔟 Feature Importance Analysis")

    if not os.path.exists(MODEL_PATH):

        print(
            "⚠️ Model not found:"
        )

        print(MODEL_PATH)

        return

    if not os.path.exists(FEATURE_NAMES_PATH):

        print(
            "⚠️ Feature names not found:"
        )

        print(FEATURE_NAMES_PATH)

        return

    model = joblib.load(
        MODEL_PATH
    )

    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "⚠️ Model does not support feature importance."
        )

        return

    importances = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importances

    }).sort_values(
        "importance",
        ascending=False
    )

    # Top 15 features
    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=top_features,
        x="importance",
        y="feature"
    )

    plt.title(
        "Top 15 Important Features - Random Forest"
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "10_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. BIAS-AWARE DEMOGRAPHIC ANALYSIS
# ============================================================

def demographic_analysis(df):

    print("\n1️⃣1️⃣ Bias-Aware Demographic Analysis")

    if "gender" not in df.columns:

        print(
            "⚠️ Gender column not found."
        )

        return

    summary = (
        df.groupby("gender")
        ["placement_numeric"]
        .mean()
        .mul(100)
        .reset_index(
            name="placement_rate"
        )
    )

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=summary,
        x="gender",
        y="placement_rate"
    )

    plt.title(
        "Placement Rate by Gender"
    )

    plt.xlabel(
        "Gender"
    )

    plt.ylabel(
        "Placement Rate (%)"
    )

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "11_demographic_analysis.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("🚀 JOB ACCEPTANCE PREDICTION SYSTEM")
    print("📊 ANALYST TASKS - EDA")
    print("=" * 70)

    # Load data
    df = load_data()

    # Prepare target
    df = prepare_target(df)

    # --------------------------------------------------------
    # Run the 11 required Analyst Tasks
    # --------------------------------------------------------

    academic_scores_vs_placement(df)

    skills_match_vs_interview(df)

    certification_impact(df)

    company_tier_acceptance(df)

    experience_vs_placement(df)

    competition_impact(df)

    interview_vs_probability(df)

    employability_test_analysis(df)

    dropout_risk_identification(df)

    feature_importance_analysis()

    demographic_analysis(df)

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)


    print("\n📁 Plots saved in:")
    print(PLOTS_DIR)

    print("\nGenerated plots:")

    print(
        "1.  Academic scores vs placement"
    )

    print(
        "2.  Skills match vs interview performance"
    )

    print(
        "3.  Certification impact"
    )

    print(
        "4.  Company tier vs acceptance"
    )

    print(
        "5.  Experience vs placement"
    )

    print(
        "6.  Competition level vs acceptance"
    )

    print(
        "7.  Interview score vs placement probability"
    )

    print(
        "8.  Employability test score analysis"
    )

    print(
        "9.  Dropout risk identification"
    )

    print(
        "10. Feature importance"
    )

    print(
        "11. Bias-aware demographic analysis"
    )

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()