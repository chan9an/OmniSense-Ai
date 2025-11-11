# --- ADD THIS LINE FIRST ---
# Disables the GPU to fix the cuDNN error
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ---------------------------

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import pickle  # <-- Import pickle to load the scaler

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load your trained model AND the scaler ---
model = None
scaler = None

try:
    # Load the Model
    model = tf.keras.models.load_model("harmodel/har_model_1D_CNN.h5")
    print("✅ Model har_model_1D_CNN.h5 loaded successfully!")
    
    # Load the Scaler
    with open('harmodel/har_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler har_scaler.pkl loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")


# --- 3. Define the labels ---
LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
WINDOW_SIZE = 80 # This must match your model

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler is not loaded"}), 500

    try:
        # Get the (80, 3) raw data window from the client
        data = request.json
        sensor_window = data.get('readings') 

        if not sensor_window or len(sensor_window) != WINDOW_SIZE:
            return jsonify({"error": f"Expected {WINDOW_SIZE} readings, but got {len(sensor_window)}"}), 400

        # --- THIS IS THE CRITICAL FIX ---
        # 1. Convert to numpy array
        sensor_window_np = np.array(sensor_window)
        
        # 2. Apply the scaler to the (80, 3) raw data
        scaled_window = scaler.transform(sensor_window_np)
        # ----------------------------------------

        # 3. Reshape the SCALED data for the model
        model_input = scaled_window.reshape(1, WINDOW_SIZE, 3)

        # --- Make the prediction ---
        prediction_scores = model.predict(model_input)
        predicted_index = np.argmax(prediction_scores)
        predicted_activity = LABELS[predicted_index]
        confidence = float(np.max(prediction_scores))

        # Print to your server terminal
        print(f"Prediction: {predicted_activity} (Confidence: {confidence:.2f})")

        # Send the result back to the client
        return jsonify({
            "activity": predicted_activity,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    print("... (GPU is disabled, running on CPU) ...")
    app.run(host="127.0.0.1", port=5000, debug=False)