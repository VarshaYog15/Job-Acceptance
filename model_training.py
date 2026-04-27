"""
MODEL TRAINING - Job Acceptance Project
--------------------------------------
✔ Train model
✔ Evaluate performance
✔ Save predictions & metrics
"""

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