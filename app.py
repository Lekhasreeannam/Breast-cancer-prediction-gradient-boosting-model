import streamlit as st
import pandas as pd
import pickle

# ==========================
# Load Saved Objects
# ==========================
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("features.pkl", "rb") as file:
    feature_names = pickle.load(file)

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")

st.write(
    """
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained **Gradient Boosting Classifier**.
"""
)

st.markdown("---")

# ==========================
# Input Fields
# ==========================

col1, col2 = st.columns(2)

user_input = {}

for i, feature in enumerate(feature_names):

    with (col1 if i % 2 == 0 else col2):
        user_input[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=0.0,
            step=0.01,
            format="%.4f"
        )

st.markdown("---")

# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    # Keep feature order exactly as training
    input_df = input_df[feature_names]

    # Scale data
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]


    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🔴 Malignant Tumor")
    else:
        st.success("🟢 Benign Tumor")

    st.markdown("---")



st.markdown("---")

st.info(
    "Model: Gradient Boosting Classifier | Dataset: Breast Cancer Wisconsin Dataset"
)