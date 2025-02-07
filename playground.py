import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the dataset (Ensure you are using the correct file format)
df = pd.read_csv("df.csv")  # If you saved it as CSV
# df = pd.read_pickle("df.pkl")  # If you saved it as Pickle

# Load the saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Apply the same transformation
df_scaled = scaler.transform(df)

print("Data successfully loaded and scaled!")


