import streamlit as st
import joblib
import numpy as np

model = joblib.load("typhoid_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Typhoid Prediction App")

year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
week = st.number_input("Week", min_value=1, max_value=52, value=42)
uganda_cases = st.number_input("Cases in uganda", value=990)
rainfall = st.number_input("Rainfall (mm)", value=84.83)
temperature = st.number_input("Temperature (°C)", value=28.05)
humidity = st.number_input("Humidity (%)", value=69.95)

if st.button("Predict"):
    X = np.array([[year, week, uganda_cases, rainfall, temperature, humidity]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.success(f"Predicted Typhoid Cases in Kases: {pred:.1f}")
