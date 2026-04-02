# ============================================
# Streamlit App - Customer Churn Prediction
# ============================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------- Title ----------
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on behavior")

# ---------- Load Dataset ----------
df = pd.read_csv("data/customer_churn_data.csv")

# ---------- Data Cleaning ----------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.mean(numeric_only=True), inplace=True)

# ---------- Feature Engineering ----------
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["is_long_term"] = (df["tenure"] > 24).astype(int)

# ---------- Target ----------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------- Drop ID ----------
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# ---------- Encode ----------
categorical_cols = df.select_dtypes(include=["object"]).columns
encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ---------- Split ----------
X = df.drop(columns=["Churn"])
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Train Model ----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ============================================
# USER INPUT SECTION
# ============================================

st.sidebar.header("Enter Customer Details")

def user_input():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 10, 150, 70)
    total_charges = st.sidebar.slider("Total Charges", 0, 10000, 1000)

    contract = st.sidebar.selectbox("Contract", encoders["Contract"].classes_)
    payment_method = st.sidebar.selectbox("Payment Method", encoders["PaymentMethod"].classes_)
    internet_service = st.sidebar.selectbox("Internet Service", encoders["InternetService"].classes_)

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service
    }

    return pd.DataFrame([data])

input_df = user_input()

# ---------- Preprocess Input ----------
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Add engineered features
input_df["avg_monthly_spend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
input_df["is_long_term"] = (input_df["tenure"] > 24).astype(int)

# Match training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# ---------- Prediction ----------
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ---------- Output ----------
st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2f})")
else:
    st.success(f"✅ Customer is likely to STAY (Probability: {probability:.2f})")

# Show input
st.subheader("Input Data")
st.write(input_df)