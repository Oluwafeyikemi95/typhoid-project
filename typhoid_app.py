import streamlit as st
import joblib
import numpy as np
import gdown
import os

# -----------------------------
# Download model if not exists
# -----------------------------
model_path = "typhoid_model.pkl"
file_id = "1NWIeM9_wRakpy3h3FqT9RnXJ6mkbbl7D"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load(model_path)
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Feature setup
# -----------------------------
expected_features = scaler.n_features_in_
st.write(f"Scaler expects {expected_features} features.")

# Main 6 features (user input)
year = st.number_input("Year", value=2025)
week = st.number_input("Week", value=42)
uganda_cases = st.number_input("Cases in Uganda", value=990)
rainfall = st.number_input("Rainfall (mm)", value=84.83)
temperature = st.number_input("Temperature (°C)", value=28.05)
humidity = st.number_input("Humidity (%)", value=69.95)

input_data = [year, week, uganda_cases, rainfall, temperature, humidity]

# Fill remaining features with realistic defaults
remaining_features = expected_features - len(input_data)
if remaining_features > 0:
    # You can adjust these defaults based on typical values
    default_values = [50.0, 70.0, 0.8]  # Example: mobility index, sanitation score, vaccination rate
    input_data += default_values[:remaining_features]

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
        st.success(f"Predicted Typhoid Cases in Kases: {pred:.1f}")



