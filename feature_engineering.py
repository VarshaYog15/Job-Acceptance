import pandas as pd

from config import CLEAN_DATA_PATH, FEATURE_DATA_PATH


def feature_engineering(df):
    """
    Perform feature engineering on the cleaned job placement dataset.

    Returns:
        pd.DataFrame: Feature-engineered dataset
    """

    print("\n🔧 FEATURE ENGINEERING STARTED")

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    # =========================================================
    # 1. INTERVIEW SCORE AVERAGE
    # =========================================================

    required_interview_columns = [
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    for col in required_interview_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing from the dataset."
            )

    df["interview_score_avg"] = (
        df["technical_score"]
        + df["aptitude_score"]
        + df["communication_score"]
    ) / 3

    print("✅ Interview score average created")

    # =========================================================
    # 2. EXPERIENCE LEVEL
    # =========================================================

    if "years_of_experience" not in df.columns:
        raise ValueError(
            "Required column 'years_of_experience' is missing."
        )

    df["experience_level"] = pd.cut(
        df["years_of_experience"],
        bins=[-1, 1, 5, float("inf")],
        labels=["fresher", "junior", "senior"],
        include_lowest=True
    )

    print("✅ Experience level created")

    # =========================================================
    # 3. SKILLS MATCH LEVEL
    # =========================================================

    if "skills_match_percentage" not in df.columns:
        raise ValueError(
            "Required column 'skills_match_percentage' is missing."
        )

    df["skills_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[-0.01, 40, 70, 100],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    print("✅ Skills match level created")

    # =========================================================
    # 4. ACADEMIC AVERAGE
    # =========================================================

    academic_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage"
    ]

    for col in academic_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing."
            )

    df["academic_avg"] = (
        df["ssc_percentage"]
        + df["hsc_percentage"]
        + df["degree_percentage"]
    ) / 3

    print("✅ Academic average created")

    # =========================================================
    # 5. ACADEMIC PERFORMANCE BAND
    # =========================================================

    df["academic_performance_band"] = pd.cut(
        df["academic_avg"],
        bins=[-0.01, 60, 75, 100],
        labels=["low", "medium", "high"],
        include_lowest=True
    )

    print("✅ Academic performance band created")

    # =========================================================
    # 6. INTERVIEW PERFORMANCE CATEGORY
    # =========================================================

    df["interview_performance_category"] = pd.cut(
        df["interview_score_avg"],
        bins=[-0.01, 50, 75, 100],
        labels=["poor", "average", "excellent"],
        include_lowest=True
    )

    print("✅ Interview performance category created")

    # =========================================================
    # 7. PLACEMENT PROBABILITY SCORE
    # =========================================================
    #
    # This is a rule-based analytical score.
    # It is NOT the machine-learning prediction.
    #
    # The formula follows the original project implementation:
    # Skills Match      = 40%
    # Interview Average = 30%
    # Academic Average  = 30%
    #
    # =========================================================

    df["placement_probability_score"] = (
        0.40 * df["skills_match_percentage"]
        + 0.30 * df["interview_score_avg"]
        + 0.30 * df["academic_avg"]
    ) / 100

    print("✅ Placement probability score created")

    # =========================================================
    # 8. CTC GAP
    # =========================================================

    required_ctc_columns = [
        "expected_ctc_lpa",
        "previous_ctc_lpa"
    ]

    for col in required_ctc_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing."
            )

    df["ctc_gap"] = (
        df["expected_ctc_lpa"]
        - df["previous_ctc_lpa"]
    )

    print("✅ CTC gap created")

    # =========================================================
    # 9. CERTIFICATION FLAG
    # =========================================================

    if "certifications_count" in df.columns:

        df["has_certifications"] = (
            df["certifications_count"] > 0
        ).astype(int)

        print("✅ Certification flag created")

    # =========================================================
    # 10. LONG NOTICE PERIOD FLAG
    # =========================================================

    if "notice_period_days" in df.columns:

        df["long_notice_period"] = (
            df["notice_period_days"] > 60
        ).astype(int)

        print("✅ Long notice period flag created")

    # =========================================================
    # 11. EMPLOYMENT GAP FLAG
    # =========================================================

    if "employment_gap_months" in df.columns:

        df["employment_gap_flag"] = (
            df["employment_gap_months"] > 0
        ).astype(int)

        print("✅ Employment gap flag created")

    # =========================================================
    # FINAL CHECK
    # =========================================================

    print("\n📊 Feature Engineering Completed")

    print(f"Original/Current Shape: {df.shape}")

    print("\n📋 New Features Created:")

    new_features = [
        "interview_score_avg",
        "experience_level",
        "skills_level",
        "academic_avg",
        "academic_performance_band",
        "interview_performance_category",
        "placement_probability_score",
        "ctc_gap",
        "has_certifications",
        "long_notice_period",
        "employment_gap_flag"
    ]

    for feature in new_features:
        if feature in df.columns:
            print(f"   ✓ {feature}")

    return df


def main():

    print("🚀 FEATURE ENGINEERING SCRIPT STARTED")

    # =========================================================
    # LOAD CLEANED DATA
    # =========================================================

    print("\n📥 Loading cleaned data...")

    try:
        df = pd.read_csv(CLEAN_DATA_PATH)

    except FileNotFoundError:

        print("\n❌ ERROR: Cleaned dataset not found.")
        print(f"Expected location:")
        print(CLEAN_DATA_PATH)

        raise

    print("✅ Cleaned dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    df_features = feature_engineering(df)

    # =========================================================
    # SAVE FEATURE-ENGINEERED DATA
    # =========================================================

    df_features.to_csv(
        FEATURE_DATA_PATH,
        index=False
    )

    print("\n💾 Feature-engineered data saved to:")
    print(FEATURE_DATA_PATH)

    print("\n✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()