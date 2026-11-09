# Job Acceptance Prediction System

## Project Overview
An end-to-end machine learning project that predicts whether a job candidate is likely to be placed using candidate academic performance, technical skills, interview scores, experience, and employment-related factors.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Machine Learning Model
- Algorithm: Random Forest Classifier
- Number of estimators: 300
- Training approach: 80:20 train-test split
- SMOTE applied to training data

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 89.07% |
| Precision | 82.99% |
| Recall | 80.36% |
| F1 Score | 81.65% |
| ROC-AUC | 95.56% |

## Dashboard Features
- Candidate placement analytics
- Business KPI overview
- Exploratory data analysis
- Candidate job acceptance prediction
- Model performance evaluation
- Confusion matrix and ROC curve
- Feature importance analysis

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
python -m streamlit run dashboard.py
```

## Project Structure

- `dashboard.py` – Streamlit dashboard
- `Job_Acceptance.ipynb` – Machine learning workflow
- `config.py` – Project configuration
- `artifacts/` – Trained model and preprocessing artifacts
- `outputs/` – EDA and evaluation outputs
- `feature_engineering.py` – Feature engineering
- `model_training.py` – Model training
- `evaluation.py` – Model evaluation

## Disclaimer
This project is developed for educational and analytical purposes. Predictions are model estimates and should not be used as the sole basis for hiring decisions.