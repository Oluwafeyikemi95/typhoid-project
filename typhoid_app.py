# typhoid_app.py
import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("typhoid_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = scaler.n_features_in_

st.title("🩺 Typhoid Prediction App")
st.write(f"Scaler expects {expected_features} features.")

# -----------------------------
# User inputs
# -----------------------------
age = st.number_input("Age", value=30, min_value=0)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 0 if gender == "Male" else 1
socio_status = st.number_input("Socioeconomic Status", value=1, min_value=0)
water_type = st.number_input("Water Source Type", value=1, min_value=0)
blood_culture = st.number_input("Blood Culture Result", value=0, min_value=0)
widal_test = st.number_input("Widal Test", value=0, min_value=0)
typhidot_test = st.number_input("Typhidot Test", value=0, min_value=0)
vaccination_status = st.number_input("Typhoid Vaccination Status", value=0, min_value=0)
weather_condition = st.number_input("Weather Condition", value=1, min_value=0)

# Combine features
input_data = [
    age, gender_val, socio_status, water_type, blood_culture,
    widal_test, typhidot_test, vaccination_status, weather_condition
]
X = np.array([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if X.shape[1] != expected_features:
        st.error(f"Input has {X.shape[1]} features, but scaler expects {expected_features}.")
    else:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Typhoid Status: {pred}")




