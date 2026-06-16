# imports
import sklearn
import xgboost
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import os

# load model & scaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes to project root
model_path = os.path.join(BASE_DIR, "models","churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models","scaler.pkl")
model_columns_path = os.path.join(BASE_DIR, "models","model_columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(model_columns_path)

# title
st.title("Customer Churn Predictor")

st.write(
    """
    Predict whether a customer is likely to churn based on account and service information.
    """
)

# input fields

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senio Citizen",
      ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    24
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value = 0.0,
    value = 70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value = 0.0,
    value = 800.0
)


if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen" : [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # one hot encoding
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Align columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # scale features
    input_scaled = scaler.transform(
        input_data
    )

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    threshold = 0.35
    prediction = 1 if probability >= threshold else 0

    if probability >= 0.60:
        st.error(
            "High Risk Customer"
        )

    elif probability >= 0.35:
        st.warning(
            "Medium Risk Customer"
        )

    else:
        st.success(
            "Low Risk Customer"
        )

    st.metric("Churn Probability", f"{probability:.1%}")    

    if prediction == 1:
        st.error(
            f"Prediction: Customer Likely To Churn ({probability:.1%})"
        )

    else:
        st.success(
            f"Prediction: Customer Likely To Stay ({ 1 - probability:.1%})"
        )