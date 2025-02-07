import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained GMM model (which includes the scaler)
with open("gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

scaler = gmm.scaler  # Extract the attached scaler

# Streamlit UI
st.title("Gaussian Mixture Model Prediction App")
st.write("Enter feature values to predict the cluster.")

# User input fields
feature1 = st.number_input("Feature 1", min_value=-10.0, max_value=10.0, step=0.1)
feature2 = st.number_input("Feature 2", min_value=-10.0, max_value=10.0, step=0.1)

# Convert input into a DataFrame
user_data = pd.DataFrame([[feature1, feature2]], columns=["Feature1", "Feature2"])

# Model internally scales the data before prediction
user_data_scaled = scaler.transform(user_data)

# Predict the cluster
if st.button("Predict Cluster"):
    cluster = gmm.predict(user_data_scaled)[0]
    st.success(f"The predicted cluster is: **{cluster}**")
