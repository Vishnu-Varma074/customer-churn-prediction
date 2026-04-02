# ============================================
# Behavior-Driven Customer Churn Prediction
# ============================================

# ---------- Imports ----------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight


# ---------- Load Dataset ----------
df = pd.read_csv("data/customer_churn_data.csv")


# ---------- Data Cleaning ----------
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


# ---------- Feature Engineering ----------
# Behavioral proxy features
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["is_long_term"] = (df["tenure"] > 24).astype(int)


# ---------- Target Variable ----------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# ---------- Drop Unnecessary Columns ----------
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)


# ---------- Encode Categorical Variables ----------
encoders = {}

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ---------- Split Features & Target ----------
X = df.drop(columns=["Churn"])
y = df["Churn"]


# ---------- Feature Scaling ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------- Handle Class Imbalance ----------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}


# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ---------- Model Training ----------
# Logistic Regression
log_model = LogisticRegression(
    class_weight=class_weight_dict,
    max_iter=1000
)
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)
rf_model.fit(X_train, y_train)


# ---------- Evaluation Function ----------
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return precision, recall, f1, roc_auc, cm


# ---------- Logistic Regression Evaluation ----------
log_results = evaluate_classification(log_model, X_test, y_test)

print("Logistic Regression Metrics")
print("Precision:", log_results[0])
print("Recall:", log_results[1])
print("F1-score:", log_results[2])
print("ROC-AUC:", log_results[3])
print("Confusion Matrix:\n", log_results[4])
print(classification_report(y_test, log_model.predict(X_test)))


# ---------- Random Forest Evaluation ----------
rf_results = evaluate_classification(rf_model, X_test, y_test)

print("\nRandom Forest Metrics")
print("Precision:", rf_results[0])
print("Recall:", rf_results[1])
print("F1-score:", rf_results[2])
print("ROC-AUC:", rf_results[3])
print("Confusion Matrix:\n", rf_results[4])
print(classification_report(y_test, rf_model.predict(X_test)))


# ---------- Feature Importance ----------
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)


import joblib

# Save model
joblib.dump(rf_model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save encoders
joblib.dump(encoders, "encoders.pkl")

# Save feature columns (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "columns.pkl")
