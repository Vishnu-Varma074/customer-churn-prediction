# ============================================
# Streamlit App - Customer Churn Prediction
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load Saved Files ----------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

# ---------- App Title ----------
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn based on input features")

# ============================================
# USER INPUT
# ============================================

st.sidebar.header("Enter Customer Details")

def user_input():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 10, 150, 70)
    total_charges = st.sidebar.slider("Total Charges", 0, 10000, 1000)

    contract = st.sidebar.selectbox(
        "Contract", encoders["Contract"].classes_
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method", encoders["PaymentMethod"].classes_
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", encoders["InternetService"].classes_
    )

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

# ============================================
# PREPROCESS INPUT
# ============================================

# Encode categorical features
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Feature Engineering (same as training)
input_df["avg_monthly_spend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
input_df["is_long_term"] = (input_df["tenure"] > 24).astype(int)

# Ensure same column order
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# ============================================
# PREDICTION
# ============================================

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ============================================
# OUTPUT
# ============================================

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2f})")
else:
    st.success(f"✅ Customer is likely to STAY (Probability: {probability:.2f})")

# Show input data
st.subheader("Input Data")
st.write(input_df)
