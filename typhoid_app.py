import streamlit as st
import joblib
import numpy as np
import os
import gdown

# -----------------------------
# Download model and scaler if not exists
# -----------------------------
model_path = "typhoid_model.pkl"
scaler_path = "scaler.pkl"
model_file_id = "1NWIeM9_wRakpy3h3FqT9RnXJ6mkbbl7D"  # Replace with your actual file IDs
scaler_file_id = "YOUR_SCALER_FILE_ID"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={model_file_id}"
    gdown.download(url, model_path, quiet=False)

if not os.path.exists(scaler_path):
    url = f"https://drive.google.com/uc?id={scaler_file_id}"
    gdown.download(url, scaler_path, quiet=False)

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_features = scaler.n_features_in_

st.title("🩺 Typhoid Prediction App")
st.write(f"Scaler expects {expected_features} features.")

# -----------------------------
# User inputs with defaults
# -----------------------------
age = st.slider("Age", min_value=0, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 0 if gender == "Male" else 1
socio_status = st.slider("Socioeconomic Status", min_value=0, max_value=3, value=1)
water_type = st.slider("Water Source Type", min_value=0, max_value=3, value=1)
blood_culture = st.slider("Blood Culture Result", min_value=0, max_value=1, value=0)
widal_test = st.slider("Widal Test", min_value=0, max_value=2, value=0)
typhidot_test = st.slider("Typhidot Test", min_value=0, max_value=2, value=0)
vaccination_status = st.slider("Typhoid Vaccination Status", min_value=0, max_value=2, value=0)
weather_condition = st.slider("Weather Condition", min_value=0, max_value=3, value=1)

# Combine features
X = np.array([[age, gender_val, socio_status, water_type,
               blood_culture, widal_test, typhidot_test,
               vaccination_status, weather_condition]])

# -----------------------------
# Prediction mapping
# -----------------------------
status_map = {
    1: "Negative",
    2: "Positive",
    3: "Suspected"
}

if st.button("Predict"):
    if X.shape[1] != expected_features:
        st.error(f"Input has {X.shape[1]} features, but scaler expects {expected_features}.")
    else:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Typhoid Status: {status_map.get(pred, 'Unknown')}")




