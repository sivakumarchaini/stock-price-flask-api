import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage (Render has no GPU)

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Optimize TensorFlow memory usage
physical_devices = tf.config.list_physical_devices("CPU")
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")

# Load the trained model
model = tf.keras.models.load_model("stock_price_lstm_model.keras")

# Load the scaler
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Stock Price Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["stock_data"]
        data = np.array(data).reshape(1, 60, 1)
        prediction = model.predict(data)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
