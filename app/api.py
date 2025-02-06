from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and encoder
model = pickle.load(open("model/asthma_risk_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Extract input features
    input_data = pd.DataFrame([data])
    
    # One-hot encode Season
    season_encoded = encoder.transform(input_data[["Season"]])
    season_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(["Season"]))
    
    # Combine scaled numerical features and encoded season
    input_data = pd.concat([input_data.drop(columns=["Season"]), season_df], axis=1)
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    
    return jsonify({"asthma_risk_score": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
