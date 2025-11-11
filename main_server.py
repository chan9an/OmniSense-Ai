# --- ADD THIS LINE FIRST ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disables GPU for TF

# --- Flask & Server Imports ---
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
import eventlet 

# --- HAR Project Imports ---
import numpy as np
import tensorflow as tf
import pickle

# --- Audio Project Imports ---
import logging
import torch
import librosa
from pydub import AudioSegment
from werkzeug.utils import secure_filename

# --- 1. Initialize the Flask App & SocketIO ---
app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 2. Setup Logging and Upload Folder ---
logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================================================================
# LOAD ALL MODELS
# ==============================================================================

# --- 3. Load HAR Model & Scaler ---
har_model = None
har_scaler = None
try:
    har_model = tf.keras.models.load_model("har_model/har_model_1D_CNN.h5")
    with open('har_model/har_scaler.pkl', 'rb') as f:
        har_scaler = pickle.load(f)
    logging.info("✅ [HAR] Model and Scaler loaded successfully!")
except Exception as e:
    logging.error(f"❌ [HAR] Error loading model or scaler: {e}")

# --- 4. Load Audio Model & Labels ---
audio_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
audio_model = None
audio_labels = None
try:
    audio_model = torch.load("audio_model/esc50resnet.pth", map_location=audio_device, weights_only=False)
    audio_model.eval()
    with open('audio_model/indtocat.pkl', 'rb') as f:
        audio_labels = pickle.load(f)
    logging.info(f"✅ [Audio] Model and Labels loaded successfully to {audio_device}!")
except Exception as e:
    logging.error(f"❌ [Audio] Error loading model or labels: {e}")

# --- 5. Define HAR Labels ---
HAR_LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
HAR_WINDOW_SIZE = 80

# --- 6. Audio Helper Functions ---
SAMPLE_RATE = 44100
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8300
TOP_DB = 80

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    if std < eps: spec_norm = spec - mean
    else: spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    delta = spec_max - spec_min
    if delta < eps: return np.zeros(spec.shape, dtype=np.uint8)
    spec_scaled = 255 * (spec_norm - spec_min) / delta
    return spec_scaled.astype(np.uint8)

def preprocess_audio_pytorch(file_path):
    try:
        wav, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = SAMPLE_RATE * DURATION
        if wav.shape[0] < target_length:
            pad_amount = int(np.ceil((target_length - wav.shape[0]) / 2))
            wav = np.pad(wav, pad_amount, mode='reflect')
        wav = wav[:target_length]
        spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        spec_db = librosa.power_to_db(spec, top_db=TOP_DB)
        spec_img = spec_to_image(spec_db)
        spec_tensor = torch.tensor(spec_img).to(audio_device, dtype=torch.float32)
        spec_tensor = spec_tensor.reshape(1, 1, *spec_tensor.shape)
        return spec_tensor
    except Exception as e:
        logging.error(f"Error in preprocess_audio_pytorch: {e}")
        return None

# ==============================================================================
# FLASK ROUTES / ENDPOINTS
# ==============================================================================

# --- 7. Route for the Main Dashboard ---
@app.route("/")
def index():
    logging.info("Serving main dashboard (index.html)")
    return send_from_directory('.', 'index.html')

# --- 8. Route for the HAR Client (har_client.py) ---
@app.route("/predict_activity", methods=["POST"]) 
def predict_activity():
    if not har_model or not har_scaler:
        return jsonify({"error": "HAR model or scaler is not loaded"}), 500
    try:
        data = request.json
        sensor_window = data.get('readings')
        if not sensor_window or len(sensor_window) != HAR_WINDOW_SIZE:
            return jsonify({"error": f"Expected {HAR_WINDOW_SIZE} readings"}), 400

        sensor_window_np = np.array(sensor_window)
        scaled_window = har_scaler.transform(sensor_window_np)
        model_input = scaled_window.reshape(1, HAR_WINDOW_SIZE, 3)
        prediction_scores = har_model.predict(model_input, verbose=0)
        
        predicted_index = np.argmax(prediction_scores)
        predicted_activity = HAR_LABELS[predicted_index] # e.g., "Sitting"
        confidence = float(np.max(prediction_scores))

        # --- THIS IS THE FIX ---
        # Group "Sitting" and "Standing" into "Still"
        activity_to_send = predicted_activity
        if predicted_activity == "Sitting" or predicted_activity == "Standing":
            activity_to_send = "Still"
        # --- END OF FIX ---

        logging.info(f"[HAR] Prediction: {predicted_activity} -> Sending: {activity_to_send} (Conf: {confidence:.2f})")
        
        socketio.emit('har_prediction', {
            'activity': activity_to_send, # <-- Use the new variable
            'confidence': confidence
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- 9. Route for the Audio Dashboard (index.html) ---
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if not audio_model or not audio_labels:
        return jsonify({"error": "Audio model or labels not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "Empty filename"}), 400

    try:
        filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(temp_filepath)
        
        clean_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], "clean_input.wav")
        sound = AudioSegment.from_file(temp_filepath)
        sound.export(clean_wav_path, format="wav")

        input_data = preprocess_audio_pytorch(clean_wav_path)
        if input_data is None:
            return jsonify({"error": "Failed to process audio"}), 500

        with torch.no_grad():
            prediction = audio_model.forward(input_data)
        
        prediction_index = int(prediction.argmax(dim=1).cpu().item())
        confidence = float(torch.softmax(prediction, dim=1)[0, prediction_index].cpu().item())
        label = audio_labels[prediction_index]
        
        logging.info(f"[Audio] Prediction: {label} (Conf: {confidence:.2f})")

        return jsonify({"prediction": label, "confidence": round(confidence, 2)})
    except Exception as e:
        logging.exception("Audio prediction failed")
        return jsonify({"error": str(e)}), 500

# --- 10. Run the Master Server ---
if __name__ == "__main__":
    logging.info("Starting master health dashboard server on http://127.0.0.1:5000")
    logging.info("... (GPU is disabled for TensorFlow) ...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False) # Host 0.0.0.0 for phone access