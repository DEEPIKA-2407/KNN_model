import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Student Result Prediction")

# Inputs
traveltime = st.number_input("Travel Time", 1, 4)
studytime = st.number_input("Study Time", 1, 4)
failures = st.number_input("Failures", 0, 3)
absences = st.number_input("Absences", 0, 100)
health = st.number_input("Health", 1, 5)

if st.button("Predict"):
    try:
        input_data = np.array([[traveltime, studytime, failures, absences, health]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.write("Prediction:", prediction[0])

    except Exception as e:
        st.error(f"Error: {e}")
        
