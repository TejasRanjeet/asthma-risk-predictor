import requests
import streamlit as st

API_URL = "https://web-production-00d6f.up.railway.app/predict"  # Update this!

st.title("Asthma Risk Prediction")

AQI = st.number_input("Enter AQI", min_value=0, max_value=500, value=100)
PM2_5 = st.number_input("Enter PM2.5", min_value=0, max_value=500, value=50)
PM10 = st.number_input("Enter PM10", min_value=0, max_value=500, value=80)
Humidity = st.number_input("Enter Humidity", min_value=0, max_value=100, value=60)
Temperature = st.number_input("Enter Temperature (°C)", min_value=-30, max_value=50, value=25)
Season = st.selectbox("Select Season", ["Winter", "Spring", "Summer", "Autumn"])

if st.button("Predict Asthma Risk"):
    input_data = {
        "AQI": AQI,
        "PM2.5": PM2_5,
        "PM10": PM10,
        "Humidity": Humidity,
        "Temperature": Temperature,
        "Season": Season
    }

    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Asthma Risk Score: {result['asthma_risk_score']}")
    else:
        st.error("Error in prediction request!")
