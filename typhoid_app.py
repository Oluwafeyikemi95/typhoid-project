import streamlit as st
import joblib
import numpy as np
import gdown
import os

model_path = "typhoid_model.pkl"
file_id = "1NWIeM9_wRakpy3h3FqT9RnXJ6mkbbl7D"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)
scaler = joblib.load("scaler.pkl")

expected_features = scaler.n_features_in_

st.title("🩺 Typhoid Prediction App")

st.write(f"Scaler expects {expected_features} feature(s).")

year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
week = st.number_input("Week", min_value=1, max_value=52, value=42)
uganda_cases = st.number_input("Cases in Uganda", value=990)
rainfall = st.number_input("Rainfall (mm)", value=84.83)
temperature = st.number_input("Temperature (°C)", value=28.05)
humidity = st.number_input("Humidity (%)", value=69.95)

X = np.array([[year, week, uganda_cases, rainfall, temperature, humidity]])

if X.shape[1] != expected_features:
    st.error(f"Error: Input has {X.shape[1]} features, but scaler expects {expected_features}.")
else:
    if st.button("Predict"):
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Typhoid Cases in Kases: {pred:.1f}")


