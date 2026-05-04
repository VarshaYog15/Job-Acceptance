
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# FILE PATH
# --------------------------------------------------
DATA_PATH = r"C:/Users/2SIN/Documents/Python/venv/Job_Acceptance/job_acceptance_features.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# TARGET VARIABLE PREPARATION
# --------------------------------------------------
# Convert 'status' → numeric
df["placement_numeric"] = df["status"].map({
    "placed": 1,
    "not placed": 0
})


# --------------------------------------------------
# DROP UNUSED COLUMNS
# --------------------------------------------------
X = df.drop(columns=["status", "placement_numeric"], errors="ignore")
y = df["placement_numeric"]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Data scaled")


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
model = LogisticRegression(max_iter=2000)

print("⚙️ Training model...")
model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
y_pred = model.predict(X_test)

print("🔮 Prediction completed")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results_df.to_csv("model_results.csv", index=False)

print("💾 Predictions saved → model_results.csv")


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------
with open("model_metrics.txt", "w") as f:
    f.write("MODEL PERFORMANCE\n")
    f.write("-----------------\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

print("💾 Metrics saved → model_metrics.txt")


print("\n🎉 MODEL TRAINING COMPLETED")

# ---------------------------------------------
# STEP 7: MACHINE LEARNING MODELING
# JOB ACCEPTANCE PREDICTION
# ---------------------------------------------

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------
# 1️⃣ Load Dataset (SAFE PATH HANDLING)
# ---------------------------------------------

print("🔹 Model Training Started")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "job_acceptance_features.csv")

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully")

# ---------------------------------------------
# 2️⃣ Target Variable Definition
# ---------------------------------------------
# Business Logic:
# Placed     → 1 (Accepted)
# Not Placed → 0 (Rejected)

df["status"] = df["status"].map({
    "placed": 1,
    "not placed": 0,
    "accepted": 1,
    "rejected": 0,
    1: 1,
    0: 0
})

# Drop rows where target is missing
df = df.dropna(subset=["status"])

# ---------------------------------------------
# 3️⃣ Feature & Target Split
# ---------------------------------------------

X = df.drop(columns=["status"])
y = df["status"]

# ---------------------------------------------
# 4️⃣ Handle Categorical Features
# ---------------------------------------------

X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------
# 5️⃣ Train-Test Split
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------
# 6️⃣ Feature Scaling
# ---------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# 7️⃣ Model Training (Business-Friendly Model)
# ---------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

print("✅ Model training completed")

# ---------------------------------------------
# 8️⃣ Model Evaluation
# ---------------------------------------------

y_pred = model.predict(X_test_scaled)

print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------
# 9️⃣ Business Insight
# ---------------------------------------------

accepted_rate = y_pred.mean() * 100
print(f"\n📌 Predicted Job Acceptance Rate: {accepted_rate:.2f}%")

print("🔹 Model Training Finished Successfully")

