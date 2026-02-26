# Typhoid Prediction System

This project implements a machine learning–based system for predicting the likelihood of typhoid infection using patient symptoms and clinical data.

# Project Overview
Typhoid fever is a serious infectious disease that requires early detection. This system applies data preprocessing and supervised machine learning techniques to assist in predicting potential typhoid cases.

# Project Structure
 `typhoid_app.py` – main application script used for making predictions
 `scaler.pkl` – trained feature scaler used during data preprocessing
 `model.pkl` – trained machine learning prediction model
 `requirements.txt` – list of required Python dependencies
 `.devcontainer/` – development container configuration (optional)

# Methodology
1. Data collection and cleaning  
2. Feature scaling using a standard scaler  
3. Training a supervised machine learning model  
4. Saving the trained model and scaler using pickle  
5. Loading the saved model for prediction in the application  

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

# How to Run the Project
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt



   model.pkl file is being saved in google drive because it's a large file
   
   https://drive.google.com/file/d/1NWIeM9_wRakpy3h3FqT9RnXJ6mkbbl7D/view?usp=sharing
