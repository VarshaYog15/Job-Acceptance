
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from config import (
    FEATURE_DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
    ARTIFACTS_DIR
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* MAIN CONTAINER */
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }


    /* MAIN TITLE */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }


    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 20px;
    }


    /* KPI CARDS */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #171b25,
            #252c3b
        );

        border: 1px solid #3f4a5e;
        border-radius: 14px;

        padding: 15px;

        min-height: 105px;

        box-shadow:
            0 4px 14px rgba(0,0,0,0.20);
    }


    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 650 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }


    /* TAB CONTAINER */
    div[data-baseweb="tab-list"] {
        display: flex;
        gap: 8px;

        background: #111827;

        padding: 8px;

        border-radius: 14px;

        border: 1px solid #3f4a5e;

        margin-top: 20px;
        margin-bottom: 25px;
    }


    /* TAB BUTTON */
    button[data-baseweb="tab"] {
        min-height: 50px;

        padding-left: 22px;
        padding-right: 22px;

        border-radius: 10px !important;

        color: #e5e7eb !important;

        background: #1f2937 !important;

        border: 1px solid #4b5563 !important;

        font-weight: 750 !important;

        font-size: 0.95rem !important;
    }


    /* TAB HOVER */
    button[data-baseweb="tab"]:hover {
        background: #334155 !important;
        color: white !important;
        border-color: #60a5fa !important;
    }


    /* ACTIVE TAB */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #2563eb !important;

        color: white !important;

        border: 1px solid #93c5fd !important;

        box-shadow:
            0 3px 12px
            rgba(37,99,235,0.40);
    }


    /* SECTION TITLES */
    .section-title {
        font-size: 1.45rem;
        font-weight: 800;

        margin-top: 8px;
        margin-bottom: 12px;
    }


    /* INFO BOX */
    .info-note {
        background: #172033;

        border-left: 4px solid #3b82f6;

        padding: 12px 16px;

        border-radius: 8px;

        margin: 10px 0 18px 0;
    }


    /* BUTTON */
    .stButton > button {
        border-radius: 10px;
        font-weight: 750;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(FEATURE_DATA_PATH):

        raise FileNotFoundError(
            f"""
Feature-engineered dataset was not found.

Expected location:

{FEATURE_DATA_PATH}

Please run:

python feature_engineering.py
"""
        )

    data = pd.read_csv(
        FEATURE_DATA_PATH
    )

    if "status" not in data.columns:

        raise ValueError(
            "The dataset does not contain the required 'status' column."
        )

    # --------------------------------------------------------
    # Target conversion
    # --------------------------------------------------------

    data["placement_numeric"] = (
        data["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "placed": 1,
                "not placed": 0,
                "not_placed": 0
            }
        )
    )

    if data["placement_numeric"].isna().any():

        invalid_values = (
            data.loc[
                data["placement_numeric"].isna(),
                "status"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unexpected status values found: {invalid_values}"
        )

    return data


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_artifacts():

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        FEATURE_NAMES_PATH
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not os.path.exists(file_path)
    ]

    if missing_files:

        raise FileNotFoundError(
            "The following model files are missing:\n\n"
            + "\n".join(missing_files)
            + "\n\nPlease run preprocessing.py and "
              "model_training.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    return (
        model,
        preprocessor,
        feature_names
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

@st.cache_data
def calculate_model_metrics():

    model, _, _ = load_model_artifacts()

    X_test_path = os.path.join(
        ARTIFACTS_DIR,
        "X_test.pkl"
    )

    y_test_path = os.path.join(
        ARTIFACTS_DIR,
        "y_test.pkl"
    )

    if not os.path.exists(X_test_path):

        return None

    if not os.path.exists(y_test_path):

        return None

    X_test = joblib.load(
        X_test_path
    )

    y_test = joblib.load(
        y_test_path
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    else:

        probabilities = predictions


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    except Exception:

        roc_auc = None


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_test": np.asarray(y_test),
        "predictions": np.asarray(predictions),
        "probabilities": np.asarray(probabilities)
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_unique_values(column):

    if column not in df.columns:

        return ["Not available"]

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if not values:

        return ["Not available"]

    return values


def add_bar_labels(ax):

    for container in ax.containers:

        try:

            labels = []

            for bar in container:

                height = bar.get_height()

                if np.isnan(height):

                    labels.append("")

                else:

                    labels.append(
                        f"{height:.1f}%"
                    )

            ax.bar_label(
                container,
                labels=labels,
                padding=3
            )

        except Exception:

            pass


# ============================================================
# CANDIDATE FEATURE ENGINEERING
# ============================================================

def create_candidate_features(candidate):

    candidate_df = pd.DataFrame(
        [candidate]
    )


    # --------------------------------------------------------
    # Interview Average
    # --------------------------------------------------------

    candidate_df["interview_score_avg"] = (

        candidate_df["technical_score"]
        +
        candidate_df["aptitude_score"]
        +
        candidate_df["communication_score"]

    ) / 3.0


    # --------------------------------------------------------
    # Experience Level
    # --------------------------------------------------------

    candidate_df["experience_level"] = pd.cut(

        candidate_df["years_of_experience"],

        bins=[
            -1,
            1,
            5,
            float("inf")
        ],

        labels=[
            "fresher",
            "junior",
            "senior"
        ],

        include_lowest=True

    ).astype(str)


    # --------------------------------------------------------
    # Skills Level
    # --------------------------------------------------------

    candidate_df["skills_level"] = pd.cut(

        candidate_df["skills_match_percentage"],

        bins=[
            -0.01,
            40,
            70,
            100
        ],

        labels=[
            "low",
            "medium",
            "high"
        ],

        include_lowest=True

    ).astype(str)


    # --------------------------------------------------------
    # Academic Average
    # --------------------------------------------------------

    candidate_df["academic_avg"] = (

        candidate_df["ssc_percentage"]
        +
        candidate_df["hsc_percentage"]
        +
        candidate_df["degree_percentage"]

    ) / 3.0


    # --------------------------------------------------------
    # Academic Performance
    # --------------------------------------------------------

    candidate_df["academic_performance_band"] = pd.cut(

        candidate_df["academic_avg"],

        bins=[
            -0.01,
            60,
            75,
            100
        ],

        labels=[
            "low",
            "medium",
            "high"
        ],

        include_lowest=True

    ).astype(str)


    # --------------------------------------------------------
    # Interview Performance
    # --------------------------------------------------------

    candidate_df["interview_performance_category"] = pd.cut(

        candidate_df["interview_score_avg"],

        bins=[
            -0.01,
            50,
            75,
            100
        ],

        labels=[
            "poor",
            "average",
            "excellent"
        ],

        include_lowest=True

    ).astype(str)


    # --------------------------------------------------------
    # Placement Probability Score
    # --------------------------------------------------------

    candidate_df["placement_probability_score"] = (

        0.40
        * candidate_df["skills_match_percentage"]

        +

        0.30
        * candidate_df["interview_score_avg"]

        +

        0.30
        * candidate_df["academic_avg"]

    ) / 100.0


    # --------------------------------------------------------
    # CTC Gap
    # --------------------------------------------------------

    candidate_df["ctc_gap"] = (

        candidate_df["expected_ctc_lpa"]
        -
        candidate_df["previous_ctc_lpa"]

    )


    # --------------------------------------------------------
    # Certification Flag
    # --------------------------------------------------------

    candidate_df["has_certifications"] = (

        candidate_df["certifications_count"] > 0

    ).astype(int)


    # --------------------------------------------------------
    # Notice Period Flag
    # --------------------------------------------------------

    candidate_df["long_notice_period"] = (

        candidate_df["notice_period_days"] > 60

    ).astype(int)


    # --------------------------------------------------------
    # Employment Gap Flag
    # --------------------------------------------------------

    candidate_df["employment_gap_flag"] = (

        candidate_df["employment_gap_months"] > 0

    ).astype(int)


    return candidate_df


# ============================================================
# PREPARE CANDIDATE DATA FOR MODEL
# ============================================================

def prepare_candidate_for_model(candidate_df):

    categorical_columns = [

        "gender",

        "degree_specialization",

        "internship_experience",

        "career_switch_willingness",

        "relevant_experience",

        "company_tier",

        "job_role_match",

        "competition_level",

        "bond_requirement",

        "layoff_history",

        "relocation_willingness",

        "experience_level",

        "skills_level",

        "academic_performance_band",

        "interview_performance_category"

    ]

    for column in categorical_columns:

        if column in candidate_df.columns:

            candidate_df[column] = (
                candidate_df[column]
                .astype(str)
            )


    # --------------------------------------------------------
    # Determine the exact columns used when the preprocessor
    # was fitted.
    # --------------------------------------------------------

    if hasattr(
        preprocessor,
        "feature_names_in_"
    ):

        expected_columns = list(
            preprocessor.feature_names_in_
        )

    else:

        expected_columns = [
            column
            for column in candidate_df.columns
            if column != "status"
        ]


    missing_columns = [

        column
        for column in expected_columns
        if column not in candidate_df.columns

    ]

    if missing_columns:

        raise ValueError(
            "The candidate input is missing the following "
            "features required by the saved preprocessor:\n\n"
            + "\n".join(missing_columns)
        )


    # --------------------------------------------------------
    # Keep exactly the same columns and order as training.
    # --------------------------------------------------------

    candidate_df = candidate_df[
        expected_columns
    ].copy()


    return candidate_df


# ============================================================
# PREDICTION
# ============================================================

def predict_candidate(candidate):

    candidate_df = create_candidate_features(
        candidate
    )

    candidate_df = prepare_candidate_for_model(
        candidate_df
    )


    # --------------------------------------------------------
    # Transform using saved preprocessing pipeline
    # --------------------------------------------------------

    processed_array = preprocessor.transform(
        candidate_df
    )


    # --------------------------------------------------------
    # Convert transformed array into DataFrame
    # with the same feature names used during training.
    # This removes the Random Forest feature-name warning.
    # --------------------------------------------------------

    processed_data = np.asarray(processed_array)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            processed_data
        )[0]
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                processed_data
            )[0]
        )

        classes = list(
            model.classes_
        )


        if 1 in classes:

            acceptance_probability = (

                probabilities[
                    classes.index(1)
                ]

                * 100

            )

        else:

            acceptance_probability = 0


        if 0 in classes:

            rejection_probability = (

                probabilities[
                    classes.index(0)
                ]

                * 100

            )

        else:

            rejection_probability = (

                100
                -
                acceptance_probability

            )

    else:

        acceptance_probability = (

            100
            if prediction == 1
            else 0

        )

        rejection_probability = (

            100
            -
            acceptance_probability

        )


    # --------------------------------------------------------
    # Placement probability score
    # --------------------------------------------------------

    placement_score = (

        candidate_df[
            "placement_probability_score"
        ].iloc[0]

        * 100

    )


    return (
        prediction,
        acceptance_probability,
        rejection_probability,
        placement_score,
        candidate_df
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    df = load_data()

    model, preprocessor, feature_names = (
        load_model_artifacts()
    )

    model_metrics = (
        calculate_model_metrics()
    )

except Exception as error:

    st.error(
        "Dashboard initialization failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎯 Job Acceptance Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Candidate placement analytics • '
    'Random Forest prediction • '
    'Model evaluation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_candidates = len(df)


placement_rate = (

    df["placement_numeric"].mean()

    * 100

)


average_interview_score = (

    df["interview_score_avg"].mean()

)


average_skills_match = (

    df["skills_match_percentage"].mean()

)


# Dataset contains placement status rather than
# a separate offer-dropout variable.

offer_dropout_rate = (

    100
    -
    placement_rate

)


# Dashboard operational definition:
# Skills match < 50 OR interview average < 60

high_risk_candidates = df[

    (
        df["skills_match_percentage"]
        < 50
    )

    |

    (
        df["interview_score_avg"]
        < 60
    )

]


high_risk_percentage = (

    len(high_risk_candidates)
    /
    total_candidates
    *
    100

)


# ============================================================
# KPI DISPLAY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📌 Key Business KPIs'
    '</div>',
    unsafe_allow_html=True
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "Total Candidates",
        f"{total_candidates:,}"
    )


with k2:

    st.metric(
        "Placement Rate",
        f"{placement_rate:.2f}%"
    )


with k3:

    st.metric(
        "Job Acceptance Rate",
        f"{placement_rate:.2f}%"
    )


with k4:

    st.metric(
        "Average Interview Score",
        f"{average_interview_score:.2f}"
    )


k5, k6, k7 = st.columns(3)


with k5:

    st.metric(
        "Average Skills Match",
        f"{average_skills_match:.2f}%"
    )


with k6:

    st.metric(
        "Offer Dropout Rate",
        f"{offer_dropout_rate:.2f}%"
    )


with k7:

    st.metric(
        "High-Risk Candidate %",
        f"{high_risk_percentage:.2f}%"
    )


st.markdown(
    """
    <div class="info-note">
    <b>Note:</b> Offer Dropout Rate is represented as
    the non-placement rate because the dataset contains
    placement status rather than a separate offer-dropout field.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

overview_tab, analytics_tab, prediction_tab, performance_tab = (

    st.tabs(
        [
            "🏠  Overview",
            "📊  Analytics",
            "🤖  Candidate Prediction",
            "📈  Model Performance"
        ]
    )

)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    st.markdown(
        '<div class="section-title">'
        '🏠 Placement Overview'
        '</div>',
        unsafe_allow_html=True
    )


    left, right = st.columns(2)


    # --------------------------------------------------------
    # Placement Outcome
    # --------------------------------------------------------

    with left:

        status_summary = (

            df[
                "status"
            ]

            .value_counts()

            .rename_axis(
                "status"
            )

            .reset_index(
                name="count"
            )

        )


        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )


        sns.barplot(
            data=status_summary,
            x="status",
            y="count",
            ax=ax
        )


        ax.set_title(
            "Placement Outcome"
        )

        ax.set_xlabel("")

        ax.set_ylabel(
            "Number of Candidates"
        )


        for container in ax.containers:

            ax.bar_label(
                container
            )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


    # --------------------------------------------------------
    # Company Tier
    # --------------------------------------------------------

    with right:

        company_summary = (

            df

            .groupby(
                "company_tier",
                observed=True
            )[
                "placement_numeric"
            ]

            .mean()

            .mul(100)

            .reset_index(
                name="acceptance_rate"
            )

        )


        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )


        sns.barplot(
            data=company_summary,
            x="company_tier",
            y="acceptance_rate",
            ax=ax
        )


        ax.set_title(
            "Acceptance Rate by Company Tier"
        )

        ax.set_xlabel(
            "Company Tier"
        )

        ax.set_ylabel(
            "Acceptance Rate (%)"
        )

        ax.set_ylim(
            0,
            100
        )


        add_bar_labels(
            ax
        )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


    # --------------------------------------------------------
    # Business Snapshot
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '💡 Business Snapshot'
        '</div>',
        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:

        st.info(
            f"**Placement:** "
            f"{placement_rate:.2f}% "
            f"of candidates are placed."
        )


    with b:

        st.info(
            f"**Skills:** "
            f"Average skills match is "
            f"{average_skills_match:.2f}%."
        )


    with c:

        st.info(
            f"**Risk:** "
            f"{high_risk_percentage:.2f}% "
            f"fall under the dashboard high-risk rule."
        )


# ============================================================
# ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.markdown(
        '<div class="section-title">'
        '📊 Selected Business Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "The dashboard intentionally shows only the most "
        "decision-useful EDA. The complete Analyst Tasks "
        "analysis is maintained separately in eda.py."
    )


    # ========================================================
    # 1. SKILLS MATCH VS INTERVIEW PERFORMANCE
    # ========================================================

    st.subheader(
        "🧩 Skills Match vs Interview Performance"
    )


    plot_df = df[
        [
            "skills_match_percentage",
            "interview_score_avg",
            "status"
        ]
    ].dropna().copy()


    # More useful ranges
    # than a raw scatter plot.

    plot_df["skills_range"] = pd.cut(

        plot_df[
            "skills_match_percentage"
        ],

        bins=[
            -0.01,
            40,
            50,
            60,
            70,
            80,
            90,
            100
        ],

        labels=[
            "<40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%"
        ],

        include_lowest=True

    )


    plot_df["interview_range"] = pd.cut(

        plot_df[
            "interview_score_avg"
        ],

        bins=[
            -0.01,
            40,
            50,
            60,
            70,
            80,
            90,
            100
        ],

        labels=[
            "<40",
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100"
        ],

        include_lowest=True

    )


    plot_df["placed_numeric"] = (

        plot_df[
            "status"
        ]

        .astype(str)

        .str.lower()

        .map(
            {
                "placed": 1,
                "not placed": 0
            }
        )

    )


    heatmap_data = (

        plot_df

        .pivot_table(
            index="interview_range",
            columns="skills_range",
            values="placed_numeric",
            aggfunc="mean",
            observed=False
        )

        * 100

    )


    fig, ax = plt.subplots(
        figsize=(12, 6)
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
        },

        ax=ax
    )


    ax.set_title(
        "Placement Rate by Skills Match and Interview Score"
    )

    ax.set_xlabel(
        "Skills Match Percentage"
    )

    ax.set_ylabel(
        "Average Interview Score"
    )


    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


    # ========================================================
    # 2. CERTIFICATION + EXPERIENCE
    # ========================================================

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Certification
    # --------------------------------------------------------

    with col1:

        certification_df = df.copy()


        certification_df[
            "certification_status"
        ] = np.where(

            certification_df[
                "certifications_count"
            ] > 0,

            "Certified",

            "No Certification"

        )


        certification_summary = (

            certification_df

            .groupby(
                "certification_status"
            )[
                "placement_numeric"
            ]

            .mean()

            .mul(100)

            .reset_index(
                name="acceptance_rate"
            )

        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.barplot(

            data=certification_summary,

            x="certification_status",

            y="acceptance_rate",

            ax=ax

        )


        ax.set_title(
            "Certification Impact on Job Acceptance"
        )

        ax.set_xlabel("")

        ax.set_ylabel(
            "Acceptance Rate (%)"
        )

        ax.set_ylim(
            0,
            100
        )


        add_bar_labels(
            ax
        )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    with col2:

        experience_summary = (

            df

            .groupby(
                "experience_level",
                observed=True
            )[
                "placement_numeric"
            ]

            .mean()

            .mul(100)

            .reset_index(
                name="placement_rate"
            )

        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.barplot(

            data=experience_summary,

            x="experience_level",

            y="placement_rate",

            ax=ax

        )


        ax.set_title(
            "Experience vs Placement Success"
        )

        ax.set_xlabel(
            "Experience Level"
        )

        ax.set_ylabel(
            "Placement Rate (%)"
        )

        ax.set_ylim(
            0,
            100
        )


        add_bar_labels(
            ax
        )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


    # ========================================================
    # 3. COMPETITION LEVEL
    # ========================================================

    st.subheader(
        "🏆 Competition Level Impact on Acceptance"
    )


    competition_summary = (

        df

        .groupby(
            "competition_level"
        )[
            "placement_numeric"
        ]

        .mean()

        .mul(100)

        .reset_index(
            name="acceptance_rate"
        )

    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.barplot(

        data=competition_summary,

        x="competition_level",

        y="acceptance_rate",

        ax=ax

    )


    ax.set_title(
        "Acceptance Rate by Competition Level"
    )

    ax.set_xlabel(
        "Competition Level"
    )

    ax.set_ylabel(
        "Acceptance Rate (%)"
    )

    ax.set_ylim(
        0,
        100
    )


    add_bar_labels(
        ax
    )


    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


    # ========================================================
    # 4. INTERVIEW SCORE VS PLACEMENT PROBABILITY
    # ========================================================

    st.subheader(
        "🎤 Interview Score vs Placement Probability"
    )


    probability_df = df[
        [
            "interview_score_avg",
            "placement_probability_score",
            "status"
        ]
    ].dropna().copy()


    probability_df[
        "probability_percent"
    ] = (

        probability_df[
            "placement_probability_score"
        ]

        * 100

    )


    probability_df[
        "interview_range"
    ] = pd.cut(

        probability_df[
            "interview_score_avg"
        ],

        bins=[
            -0.01,
            40,
            50,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            100
        ],

        labels=[
            "<40",
            "40-50",
            "50-60",
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


    probability_summary = (

        probability_df

        .groupby(
            [
                "interview_range",
                "status"
            ],
            observed=False
        )[
            "probability_percent"
        ]

        .mean()

        .reset_index()

    )


    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    sns.lineplot(

        data=probability_summary,

        x="interview_range",

        y="probability_percent",

        hue="status",

        marker="o",

        linewidth=2.5,

        markersize=7,

        ax=ax

    )


    ax.set_title(
        "Average Placement Probability by Interview Score Range"
    )

    ax.set_xlabel(
        "Average Interview Score Range"
    )

    ax.set_ylabel(
        "Placement Probability (%)"
    )

    ax.set_ylim(
        0,
        100
    )


    ax.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


# ============================================================
# CANDIDATE PREDICTION TAB
# ============================================================

with prediction_tab:

    st.markdown(
        '<div class="section-title">'
        '🤖 Candidate Job Acceptance Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Change the candidate details below. "
        "The calculated candidate metrics update immediately."
    )


    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        age_years = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=60,
            value=25,
            step=1
        )


        gender = st.selectbox(
            "Gender",
            get_unique_values(
                "gender"
            )
        )


        ssc_percentage = st.number_input(
            "SSC Percentage",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        hsc_percentage = st.number_input(
            "HSC Percentage",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        degree_percentage = st.number_input(
            "Degree Percentage",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        degree_specialization = st.selectbox(
            "Degree Specialization",
            get_unique_values(
                "degree_specialization"
            )
        )


        technical_score = st.number_input(
            "Technical Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        aptitude_score = st.number_input(
            "Aptitude Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        communication_score = st.number_input(
            "Communication Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        skills_match_percentage = st.number_input(
            "Skills Match Percentage",
            min_value=45.0,
            max_value=100.0,
            value=70.0,
            step=0.1
        )


        certifications_count = st.number_input(
            "Certifications Count",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )


        internship_experience = st.selectbox(
            "Internship Experience",
            get_unique_values(
                "internship_experience"
            )
        )


        years_of_experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5
        )


        career_switch_willingness = st.selectbox(
            "Career Switch Willingness",
            get_unique_values(
                "career_switch_willingness"
            )
        )


        relevant_experience = st.selectbox(
            "Relevant Experience",
            get_unique_values(
                "relevant_experience"
            )
        )


    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:

        previous_ctc_lpa = st.number_input(
            "Previous CTC (LPA)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )


        expected_ctc_lpa = st.number_input(
            "Expected CTC (LPA)",
            min_value=0.0,
            max_value=100.0,
            value=6.0,
            step=0.1
        )


        company_tier = st.selectbox(
            "Company Tier",
            get_unique_values(
                "company_tier"
            )
        )


        job_role_match = st.selectbox(
            "Job Role Match",
            get_unique_values(
                "job_role_match"
            )
        )


        competition_level = st.selectbox(
            "Competition Level",
            get_unique_values(
                "competition_level"
            )
        )


        bond_requirement = st.selectbox(
            "Bond Requirement",
            get_unique_values(
                "bond_requirement"
            )
        )


        notice_period_days = st.number_input(
            "Notice Period (days)",
            min_value=0.0,
            max_value=180.0,
            value=30.0,
            step=1.0
        )


        layoff_history = st.selectbox(
            "Layoff History",
            get_unique_values(
                "layoff_history"
            )
        )


        employment_gap_months = st.number_input(
            "Employment Gap (months)",
            min_value=0.0,
            max_value=120.0,
            value=0.0,
            step=1.0
        )


        relocation_willingness = st.selectbox(
            "Relocation Willingness",
            get_unique_values(
                "relocation_willingness"
            )
        )


    # ========================================================
    # LIVE CANDIDATE METRICS
    # ========================================================

    interview_score_avg = (

        technical_score
        +
        aptitude_score
        +
        communication_score

    ) / 3.0


    academic_avg = (

        ssc_percentage
        +
        hsc_percentage
        +
        degree_percentage

    ) / 3.0


    ctc_gap = (

        expected_ctc_lpa
        -
        previous_ctc_lpa

    )


    placement_probability_score = (

        0.40
        * skills_match_percentage

        +

        0.30
        * interview_score_avg

        +

        0.30
        * academic_avg

    ) / 100.0


    placement_score_percent = (

        placement_probability_score
        * 100

    )


    st.markdown(
        '<div class="section-title">'
        '📊 Live Calculated Candidate Metrics'
        '</div>',
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Interview Average",
            f"{interview_score_avg:.2f}"
        )


    with m2:

        st.metric(
            "Academic Average",
            f"{academic_avg:.2f}"
        )


    with m3:

        st.metric(
            "CTC Gap",
            f"{ctc_gap:.2f} LPA"
        )


    with m4:

        st.metric(
            "Placement Score",
            f"{placement_score_percent:.2f}%"
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Job Acceptance",
        type="primary",
        width="stretch"
    ):


        candidate = {

            "age_years":
                age_years,

            "gender":
                gender,

            "ssc_percentage":
                ssc_percentage,

            "hsc_percentage":
                hsc_percentage,

            "degree_percentage":
                degree_percentage,

            "degree_specialization":
                degree_specialization,

            "technical_score":
                technical_score,

            "aptitude_score":
                aptitude_score,

            "communication_score":
                communication_score,

            "skills_match_percentage":
                skills_match_percentage,

            "certifications_count":
                certifications_count,

            "internship_experience":
                internship_experience,

            "years_of_experience":
                years_of_experience,

            "career_switch_willingness":
                career_switch_willingness,

            "relevant_experience":
                relevant_experience,

            "previous_ctc_lpa":
                previous_ctc_lpa,

            "expected_ctc_lpa":
                expected_ctc_lpa,

            "company_tier":
                company_tier,

            "job_role_match":
                job_role_match,

            "competition_level":
                competition_level,

            "bond_requirement":
                bond_requirement,

            "notice_period_days":
                notice_period_days,

            "layoff_history":
                layoff_history,

            "employment_gap_months":
                employment_gap_months,

            "relocation_willingness":
                relocation_willingness

        }


        try:

            (
                prediction,
                acceptance_probability,
                rejection_probability,
                placement_score,
                engineered_candidate

            ) = predict_candidate(
                candidate
            )


            st.markdown("---")


            st.subheader(
                "🎯 Prediction Result"
            )


            if prediction == 1:

                st.success(
                    "✅ JOB ACCEPTED / PLACED"
                )

            else:

                st.error(
                    "❌ JOB NOT ACCEPTED / NOT PLACED"
                )


            r1, r2, r3 = st.columns(3)


            with r1:

                st.metric(
                    "Model Acceptance Probability",
                    f"{acceptance_probability:.2f}%"
                )


            with r2:

                st.metric(
                    "Model Rejection Probability",
                    f"{rejection_probability:.2f}%"
                )


            with r3:

                st.metric(
                    "Engineered Placement Score",
                    f"{placement_score:.2f}%"
                )


            st.progress(
                int(
                    round(
                        max(
                            0,
                            min(
                                100,
                                acceptance_probability
                            )
                        )
                    )
                ),
                text=(
                    "Acceptance Probability: "
                    f"{acceptance_probability:.2f}%"
                )
            )


            if acceptance_probability >= 75:

                st.info(
                    "🟢 High likelihood of job acceptance."
                )

            elif acceptance_probability >= 50:

                st.info(
                    "🟡 Moderate likelihood of job acceptance."
                )

            else:

                st.info(
                    "🔴 Low likelihood of job acceptance."
                )


            # ------------------------------------------------
            # Candidate feature display
            # ------------------------------------------------

            with st.expander(
                "🔍 View Engineered Candidate Features"
            ):

                display_candidate = (
                    engineered_candidate.T.copy()
                )

                # Convert everything to string.
                # This prevents Arrow datatype conflicts
                # such as 'female' mixed with integers.
                display_candidate = (
                    display_candidate.astype(str)
                )

                display_candidate.columns = [
                    "Candidate Value"
                ]

                st.dataframe(
                    display_candidate,
                    width="stretch"
                )


        except Exception as error:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(
                error
            )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.markdown(
        '<div class="section-title">'
        '📈 Random Forest Model Performance'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Metrics are calculated using the untouched "
        "test dataset (X_test and y_test). "
        "The dashboard does not retrain the model."
    )


    if model_metrics is None:

        st.warning(
            """
X_test.pkl or y_test.pkl was not found.

Please run:

python preprocessing.py

before opening the dashboard.
"""
        )


    else:

        # ====================================================
        # METRICS
        # ====================================================

        p1, p2, p3, p4, p5 = st.columns(5)


        with p1:

            st.metric(
                "Accuracy",
                f"{model_metrics['accuracy'] * 100:.2f}%"
            )


        with p2:

            st.metric(
                "Precision",
                f"{model_metrics['precision'] * 100:.2f}%"
            )


        with p3:

            st.metric(
                "Recall",
                f"{model_metrics['recall'] * 100:.2f}%"
            )


        with p4:

            st.metric(
                "F1 Score",
                f"{model_metrics['f1'] * 100:.2f}%"
            )


        with p5:

            if model_metrics["roc_auc"] is not None:

                st.metric(
                    "ROC-AUC",
                    f"{model_metrics['roc_auc'] * 100:.2f}%"
                )

            else:

                st.metric(
                    "ROC-AUC",
                    "N/A"
                )


        # ====================================================
        # CONFUSION MATRIX + ROC CURVE
        # ====================================================

        left, right = st.columns(2)


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        with left:

            cm = confusion_matrix(

                model_metrics["y_test"],

                model_metrics["predictions"]

            )


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            sns.heatmap(

                cm,

                annot=True,

                fmt="d",

                cmap="Blues",

                cbar=False,

                xticklabels=[
                    "Not Placed",
                    "Placed"
                ],

                yticklabels=[
                    "Not Placed",
                    "Placed"
                ],

                ax=ax

            )


            ax.set_title(
                "Confusion Matrix"
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )


            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


        # ----------------------------------------------------
        # ROC CURVE
        # ----------------------------------------------------

        with right:

            if model_metrics["roc_auc"] is not None:

                fpr, tpr, _ = roc_curve(

                    model_metrics["y_test"],

                    model_metrics["probabilities"]

                )


                fig, ax = plt.subplots(
                    figsize=(6, 5)
                )


                ax.plot(

                    fpr,

                    tpr,

                    linewidth=2.5,

                    label=(
                        f"ROC-AUC = "
                        f"{model_metrics['roc_auc']:.3f}"
                    )

                )


                ax.plot(

                    [0, 1],

                    [0, 1],

                    linestyle="--"

                )


                ax.set_title(
                    "ROC Curve"
                )

                ax.set_xlabel(
                    "False Positive Rate"
                )

                ax.set_ylabel(
                    "True Positive Rate"
                )


                ax.legend()


                plt.tight_layout()

                st.pyplot(
                    fig
                )

                plt.close(
                    fig
                )


        # ====================================================
        # MODEL SUMMARY
        # ====================================================

        st.subheader(
            "🌲 Model Summary"
        )


        s1, s2, s3 = st.columns(3)


        with s1:

            st.metric(
                "Model",
                "Random Forest"
            )


        with s2:

            st.metric(
                "Number of Trees",
                str(
                    getattr(
                        model,
                        "n_estimators",
                        "N/A"
                    )
                )
            )


        with s3:

            st.metric(
                "Processed Features",
                str(
                    len(feature_names)
                )
            )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        st.subheader(
            "📌 Top 10 Feature Importance"
        )


        if hasattr(
            model,
            "feature_importances_"
        ):


            importances = (
                model.feature_importances_
            )


            if len(importances) == len(
                feature_names
            ):


                importance_df = pd.DataFrame(

                    {
                        "Feature":
                            feature_names,

                        "Importance":
                            importances
                    }

                ).sort_values(

                    "Importance",

                    ascending=False

                )


                top_features = (

                    importance_df

                    .head(10)

                    .sort_values(
                        "Importance"
                    )

                )


                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )


                sns.barplot(

                    data=top_features,

                    x="Importance",

                    y="Feature",

                    ax=ax

                )


                ax.set_title(
                    "Top 10 Important Features"
                )

                ax.set_xlabel(
                    "Feature Importance"
                )

                ax.set_ylabel(
                    "Feature"
                )


                plt.tight_layout()

                st.pyplot(
                    fig
                )

                plt.close(
                    fig
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Job Acceptance Prediction System | "
    "Python | Pandas | Scikit-learn | "
    "Random Forest | Streamlit | MySQL"
)