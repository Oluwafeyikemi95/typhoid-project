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
# Determine expected features
# -----------------------------
expected_features = scaler.n_features_in_
st.write(f"Scaler expects {expected_features} feature(s).")

# -----------------------------
# Collect user input dynamically
# -----------------------------
input_data = []

# Define the features you already know
feature_labels = [
    "Year", "Week", "Cases in Uganda",
    "Rainfall (mm)", "Temperature (°C)", "Humidity (%)"
]

# Add inputs for known features
for label in feature_labels:
    value = st.number_input(label, value=0.0)
    input_data.append(value)

# If the scaler expects more features than we have, ask for the rest
missing_features = expected_features - len(input_data)
extra_values = []
if missing_features > 0:
    st.warning(f"{missing_features} additional feature(s) required for prediction.")
    for i in range(missing_features):
        val = st.number_input(f"Additional feature {i+1}", value=0.0)
        extra_values.append(val)

# Combine all input values
X = np.array([input_data + extra_values])

# -----------------------------
# Make prediction if button pressed
# -----------------------------
if st.button("Predict"):
    if X.shape[1] != expected_features:
        st.error(f"Error: Input has {X.shape[1]} features, but scaler expects {expected_features}.")
    else:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Typhoid Cases in Kases: {pred:.1f}")



