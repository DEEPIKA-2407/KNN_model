import streamlit as st
import pandas as pd
import pickle

# Load pickle files
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Student Performance Prediction")

st.title("🎓 Student Performance Prediction")
st.write("Enter feature values and click Predict.")

# Take input for all features
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

# Prediction
if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")