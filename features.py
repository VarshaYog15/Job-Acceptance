"""
Feature Engineering Module
--------------------------
Creates derived analytical features and saves the output
as a CSV file named `features.csv`.
"""

import os

def create_features(df, output_path="C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/features.csv"):
    """
    Creates new business-driven features from existing columns
    and saves the feature-engineered dataset.

    Parameters:
    df (pd.DataFrame): Cleaned dataset
    output_path (str): Path to save the output CSV

    Returns:
    pd.DataFrame: Feature-engineered dataset
    """

    # --------------------------------------------------
    # 1. Experience Category
    # --------------------------------------------------
    df["experience_category"] = df["years_of_experience"].apply(
        lambda x: "Fresher" if x < 1 else "Junior" if x < 5 else "Senior"
    )

    # --------------------------------------------------
    # 2. Skills Match Level
    # --------------------------------------------------
    df["skills_match_level"] = df["skills_match_percentage"].apply(
        lambda x: "Low" if x < 50 else "Medium" if x < 75 else "High"
    )

    # --------------------------------------------------
    # 3. Academic Performance Bands
    # --------------------------------------------------
    df["academic_avg"] = (
        df["ssc_percentage"] +
        df["hsc_percentage"] +
        df["degree_percentage"]
    ) / 3

    df["academic_performance_band"] = df["academic_avg"].apply(
        lambda x: "Low" if x < 60 else "Medium" if x < 75 else "High"
    )

    # --------------------------------------------------
    # 4. Interview Performance Category
    # --------------------------------------------------
    df["interview_performance"] = df["interview_score"].apply(
        lambda x: "Poor" if x < 50 else "Average" if x < 75 else "Excellent"
    )

    # --------------------------------------------------
    # 5. Placement Probability Score (Rule-Based)
    # --------------------------------------------------
    df["placement_probability_score"] = (
        0.4 * df["skills_match_percentage"] +
        0.3 * df["interview_score"] +
        0.3 * df["academic_avg"]
    ) / 100

    # --------------------------------------------------
    # Save Output CSV
    # --------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Feature-engineered data saved to: {output_path}")

    return df
