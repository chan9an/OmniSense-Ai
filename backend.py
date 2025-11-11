import os
import logging
import numpy as np
import librosa
import torch
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pydub import AudioSegment  # Make sure to 'pip install pydub'

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

# --- Configuration & Model Constants (from your PyTorch script) ---
MODEL_PATH = "esc50resnet.pth"
PKL_PATH = "indtocat.pkl"

# Preprocessing params from your rec_2_detect.py
SAMPLE_RATE = 44100
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8300
TOP_DB = 80

# --- Load Model and Labels ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = None
indtocat = None

try:
    # --- FIX: Set weights_only=False to load the model architecture ---
    # This is required for this .pth file and is safe since we trust the source.
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False) 
    model.eval()
    logging.info(f"✅ Successfully loaded model from {MODEL_PATH} to {device}")

    with open(PKL_PATH, 'rb') as f:
        indtocat = pickle.load(f)
    logging.info(f"✅ Successfully loaded labels from {PKL_PATH}")

except Exception as e:
    logging.error(f"❌ Failed to load model or labels: {e}")
    model = None
    indtocat = None

# --- Helper Functions (from your rec_2_detect.py) ---
def spec_to_image(spec, eps=1e-6):
    """Normalizes a spectrogram into an 8-bit image."""
    mean = spec.mean()
    std = spec.std()
    
    # --- FIX: Handle potential division by zero from silent clips ---
    if std < eps:
        spec_norm = spec - mean # All values will be ~0
    else:
        spec_norm = (spec - mean) / (std + eps)
    
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    delta = spec_max - spec_min
    
    # --- FIX: Handle second potential division by zero ---
    if delta < eps:
        return np.zeros(spec.shape, dtype=np.uint8) # Return empty image
        
    spec_scaled = 255 * (spec_norm - spec_min) / delta
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

# --- Preprocessing Function (EXACTLY matches your PyTorch script) ---
def preprocess_audio_pytorch(file_path):
    """Loads, processes, and converts a CLEAN audio file for the PyTorch model."""
    try:
        # 1. Load audio (now guaranteed to be a clean .wav)
        wav, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        target_length = SAMPLE_RATE * DURATION

        # 2. Pad/Truncate
        if wav.shape[0] < target_length:
            pad_amount = int(np.ceil((target_length - wav.shape[0]) / 2))
            wav = np.pad(wav, pad_amount, mode='reflect')
        wav = wav[:target_length]
        
        # 3. Create Mel Spectrogram
        spec = librosa.feature.melspectrogram(
            y=wav, 
            sr=sr, 
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )
        spec_db = librosa.power_to_db(spec, top_db=TOP_DB)
        
        # 4. Normalize
        spec_img = spec_to_image(spec_db)
        
        # 5. Reshape for PyTorch
        spec_tensor = torch.tensor(spec_img).to(device, dtype=torch.float32)
        spec_tensor = spec_tensor.reshape(1, 1, *spec_tensor.shape) 
        
        return spec_tensor

    except Exception as e:
        logging.error(f"Error in preprocess_audio_pytorch: {e}")
        return None

# --- Routes ---
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or indtocat is None:
            return jsonify({"error": "Model or labels not loaded"}), 500
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        # Save the uploaded file (e.g., "audio.webm")
        temp_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(temp_filepath)
        logging.info(f"📂 Saved temporary file: {temp_filepath}")

        # --- FIX: Convert the uploaded file to a clean .wav ---
        clean_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], "clean_input.wav")
        try:
            sound = AudioSegment.from_file(temp_filepath)
            sound.export(clean_wav_path, format="wav")
            logging.info(f"✅ Converted to clean WAV: {clean_wav_path}")
        except Exception as e:
            logging.error(f"❌ Failed to convert audio: {e}")
            return jsonify({"error": "Failed to convert audio file"}), 500
        # --- End of Fix ---

        # --- Process the CLEAN file ---
        input_data = preprocess_audio_pytorch(clean_wav_path)
        
        if input_data is None:
            return jsonify({"error": "Failed to process audio"}), 500

        logging.info(f"🔍 Input shape to model: {input_data.shape}")

        # --- Make prediction (PyTorch way) ---
        with torch.no_grad():
            prediction = model.forward(input_data)
        
        prediction_index = int(prediction.argmax(dim=1).cpu().item())
        confidence = float(torch.softmax(prediction, dim=1)[0, prediction_index].cpu().item())
        label = indtocat[prediction_index]
        
        logging.info(f"✅ Prediction: {label}, Confidence: {confidence:.2f}")

        # Check if the prediction is one of your targets
        if label == "coughing":
            logging.info("🎉 Cough detected!")
        elif label == "sneezing":
            logging.info("🎉 Sneeze detected!")

        return jsonify({"prediction": label, "confidence": round(confidence, 2)})

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("static"):
        logging.error("❌ 'static' folder not found. Please create it and place index.html inside.")
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)