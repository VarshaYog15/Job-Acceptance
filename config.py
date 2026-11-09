
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "HR_Job_Placement_Dataset.csv"
)


CLEAN_DATA_PATH = os.path.join(
    BASE_DIR,
    "cleaned_data.csv"
)


FEATURE_DATA_PATH = os.path.join(
    BASE_DIR,
    "feature_engineered_data.csv"
)


ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "job_acceptance_model.pkl"
)


PREPROCESSOR_PATH = os.path.join(
    ARTIFACTS_DIR,
    "preprocessor.pkl"
)


FEATURE_NAMES_PATH = os.path.join(
    ARTIFACTS_DIR,
    "feature_names.pkl"
)


TARGET_COLUMN = "status"

# Target mapping
TARGET_MAPPING = {
    "Not Placed": 0,
    "Placed": 1
}

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Random Forest settings
N_ESTIMATORS = 300

OUTPUTS_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

EDA_OUTPUT_DIR = os.path.join(
    OUTPUTS_DIR,
    "eda"
)

PLOTS_DIR = os.path.join(
    OUTPUTS_DIR,
    "plots"
)

EVALUATION_OUTPUT_DIR = os.path.join(
    OUTPUTS_DIR,
    "evaluation"
)


os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)