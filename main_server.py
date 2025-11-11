import os, logging, numpy as np, pickle, pandas as pd, torch, librosa, tensorflow as tf
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from pydub import AudioSegment
from werkzeug.utils import secure_filename

# -----------------------------------------------------
# Setup
# -----------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------
# HAR Model
# -----------------------------------------------------
HAR_LABELS = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
HAR_WINDOW_SIZE = 80
HAR_FEATURE_NAMES = ["x-axis", "y-axis", "z-axis"]

try:
    har_model = tf.keras.models.load_model("har_model/har_model_1D_CNN.h5")
    with open("har_model/har_scaler.pkl", "rb") as f:
        har_scaler = pickle.load(f)
    logging.info("✅ [HAR] Loaded")
except Exception as e:
    har_model, har_scaler = None, None
    logging.error(f"❌ [HAR] load fail: {e}")

# -----------------------------------------------------
# Audio Model
# -----------------------------------------------------
SAMPLE_RATE, DURATION = 44100, 5
N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, TOP_DB = 2048, 512, 128, 20, 8300, 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    audio_model = torch.load("audio_model/esc50resnet.pth", map_location=device, weights_only=False)
    audio_model.eval()
    with open("audio_model/indtocat.pkl", "rb") as f:
        audio_labels = pickle.load(f)
    logging.info("✅ [Audio] Loaded")
except Exception as e:
    audio_model, audio_labels = None, None
    logging.error(f"❌ [Audio] load fail: {e}")

def spec_to_image(spec, eps=1e-6):
    mean, std = spec.mean(), spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    return (255 * (spec_norm - spec_min) / (spec_max - spec_min + eps)).astype(np.uint8)

def preprocess_audio(path):
    try:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE)
        if len(wav) < SAMPLE_RATE * DURATION:
            wav = np.pad(wav, (0, SAMPLE_RATE * DURATION - len(wav)))
        spec = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH, n_mels=N_MELS,
                                              fmin=FMIN, fmax=FMAX)
        db = librosa.power_to_db(spec, top_db=TOP_DB)
        img = spec_to_image(db)
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
        return tensor
    except Exception as e:
        logging.error(f"[Audio] preprocess error: {e}")
        return None

@app.route("/predict_activity", methods=["POST"])
def predict_activity():
    try:
        data = request.json
        arr = np.array(data["readings"])
        scaled = har_scaler.transform(pd.DataFrame(arr, columns=HAR_FEATURE_NAMES))
        X = scaled.reshape(1, HAR_WINDOW_SIZE, 3)
        y = har_model.predict(X, verbose=0)
        idx = int(np.argmax(y)); conf = float(np.max(y))
        label = HAR_LABELS[idx]
        if label in ["Sitting", "Standing"]: label = "Still"
        socketio.emit("har_prediction", {"activity": label, "confidence": conf})
        logging.info(f"[HAR] {label} ({conf:.2f})")
        return jsonify({"activity": label, "confidence": conf})
    except Exception as e:
        logging.error(f"[HAR] error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    try:
        if not audio_model or not audio_labels:
            return jsonify({"error": "Audio model missing"}), 500
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        f = request.files["file"]
        temp = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(temp)
        logging.info(f"📩 File received: {f.filename}")
        sound = AudioSegment.from_file(temp)
        clean = os.path.join(UPLOAD_FOLDER, "clean.wav")
        sound.export(clean, format="wav")
        tensor = preprocess_audio(clean)
        if tensor is None:
            return jsonify({"error": "Preprocess failed"}), 500
        with torch.no_grad():
            out = audio_model(tensor)
        idx = int(out.argmax(dim=1).cpu().item())
        conf = float(torch.softmax(out, dim=1)[0, idx].cpu().item())
        label = audio_labels[idx]
        logging.info(f"🎧 [Audio] {label} ({conf:.2f})")
        socketio.emit("audio_prediction", {"label": label, "confidence": conf})
        return jsonify({"prediction": label, "confidence": conf})
    except Exception as e:
        logging.error(f"[Audio] route error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("🚀 Flask (HAR + Audio)")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
