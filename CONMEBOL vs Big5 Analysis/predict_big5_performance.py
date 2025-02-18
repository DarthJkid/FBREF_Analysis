import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model and scalers
model = tf.keras.models.load_model("brazil_to_big5_model.h5")
scaler_X = np.load("scaler_X.npy", allow_pickle=True)
scaler_y = np.load("scaler_y.npy", allow_pickle=True)

# Define performance metrics used in training
performance_columns = ["Gls/90", "G/Sh", "G/SoT", "SoT%", "SoT/90", "Sh/90", "G-PK/90", "PK/90", "PKatt/90"]

# Function to get stats from user input
def get_player_stats():
    print("Enter the player's performance stats from Brazil Serie A:")
    stats = []
    for metric in performance_columns:
        value = float(input(f"{metric}: "))
        stats.append(value)
    return np.array(stats).reshape(1, -1)

# Function to make a prediction
def predict_performance(stats):
    stats_scaled = (stats - scaler_X.min()) / (scaler_X.max() - scaler_X.min())  # Apply the same scaling
    predicted_diff_scaled = model.predict(stats_scaled)
    predicted_diff = (predicted_diff_scaled * (scaler_y.max() - scaler_y.min())) + scaler_y.min()  # Reverse scaling
    predicted_big5_stats = stats + (predicted_diff * stats / 100)  # Adjust based on percentage difference
    return predicted_big5_stats

# Main function
def main():
    stats = get_player_stats()
    predicted_stats = predict_performance(stats)
    
    print("
Predicted performance in a Big 5 league:")
    for metric, value in zip(performance_columns, predicted_stats[0]):
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
