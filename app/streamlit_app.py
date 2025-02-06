import streamlit as st
import requests

st.title("Asthma Risk Prediction")

AQI = st.number_input("AQI Level", min_value=0, max_value=500, value=100)
PM25 = st.number_input("PM2.5 Level", min_value=0, max_value=500, value=50)
PM10 = st.number_input("PM10 Level", min_value=0, max_value=500, value=50)
Humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
Temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
Season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

if st.button("Predict Risk"):
    input_data = {
        "AQI": AQI,
        "PM2.5": PM25,
        "PM10": PM10,
        "Humidity": Humidity,
        "Temperature": Temperature,
        "Season": Season
    }
    
    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Asthma Risk Score: {result['asthma_risk_score']:.2f}")
    else:
        st.error("Error in prediction")
