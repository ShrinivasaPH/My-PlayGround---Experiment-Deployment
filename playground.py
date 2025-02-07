import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the trained GMM model
with open("gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

# Streamlit UI
st.title("Gaussian Mixture Model Prediction App")
st.write("Enter feature values to predict the cluster.")

# User input fields
feature1 = st.slider("Feature 1", min_value=-10.0, max_value=10.0, step=0.1)
feature2 = st.slider("Feature 2", min_value=-10.0, max_value=10.0, step=0.1)

# Convert input into a DataFrame
user_data = pd.DataFrame([[feature1, feature2]], columns=["Feature1", "Feature2"])

# Scale the user input
user_data_scaled = scaler.transform(user_data)

# Predict the cluster
if st.button("Predict Cluster"):
    cluster = gmm.predict(user_data_scaled)[0]
    st.success(f"The predicted cluster is: **{cluster}**")

st.divider()

st.image("Cluster image.png", caption="The Clusters")