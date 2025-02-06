import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Selecting relevant features and target variable
    features = ["AQI", "PM2.5", "PM10", "Humidity", "Temperature", "Season"]
    target = "Risk_Score"

    X = df[features]
    y = df[target]

    # Encoding categorical feature (Season)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    season_encoded = encoder.fit_transform(X[["Season"]])

    # Creating a DataFrame for encoded seasons
    season_cols = encoder.get_feature_names_out(["Season"])
    season_df = pd.DataFrame(season_encoded, columns=season_cols, index=X.index)

    # Concatenating encoded season data
    X = pd.concat([X.drop(columns=["Season"]), season_df], axis=1)

    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler, encoder, and preprocessed data
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("model/preprocessed_data.pkl", "wb") as f:
        pickle.dump((X_scaled, y), f)

    return X_scaled, y

if __name__ == "__main__":
    file_path = "data/asthma_data.csv"
    X, y = load_and_preprocess_data(file_path)
    print("Preprocessing complete. Data is ready for training.")
