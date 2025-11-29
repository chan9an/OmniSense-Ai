````markdown
# 🩺 OmniSense AI — Multi-Modal Activity & Health Monitoring

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-green)
![React](https://img.shields.io/badge/Frontend-React-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📘 Overview

OmniSense AI is an end-to-end multi-modal intelligence system designed for real-time **Human Activity Recognition (HAR)** and **respiratory event detection**.

The platform integrates motion-sensor time-series data with audio spectral analysis to classify activities and detect health events such as coughing or sneezing. It includes a full pipeline: preprocessing, model training, a low-latency Flask backend, and a responsive React dashboard.

---

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Deep Learning** | Keras (CNN), TensorFlow |
| **Machine Learning** | scikit-learn (Random Forest), NumPy, Pandas |
| **Backend API** | Python (Flask), REST architecture |
| **Frontend** | React, JavaScript, CSS Modules |
| **Signal & Audio Processing** | Librosa, SciPy |

---

## ✨ Key Features

### 🔹 Multi-Modal Deep Learning  
Supports two synchronized data streams:
- **Motion:** 1D-CNN for classifying activities (walking, running, standing)  
- **Audio:** Spectrogram analysis to detect respiratory events  

### 🔹 Model Benchmarking  
Includes performance comparisons between CNN models and Random Forest baselines.

### 🔹 Real-Time Backend  
Optimized Flask API with **<200ms** latency for real-time inference.

### 🔹 Interactive Dashboard  
React interface featuring:
- Live activity predictions  
- Respiratory event detection  
- Real-time visualizations and metrics  

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js & npm

---

## 1. Clone the Repository

```bash
git clone https://github.com/chan9an/OmniSense-AI.git
cd OmniSense-AI
```

---

## 2. Backend Setup (Flask)

```bash
cd server

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python app.py
```

**Backend running at:** http://localhost:5000

---

## 3. Frontend Setup (React)

```bash
cd client
npm install
npm start
```

**Frontend running at:** http://localhost:3000

---

## 📁 Project Structure

```text
OmniSense-AI/
├── client/                 # React frontend
│   ├── src/components/     # UI components & visualizations
│   └── public/
├── server/                 # Flask backend
│   ├── models/             # Saved .h5 / .pkl models
│   ├── processing/         # Audio / motion feature extraction
│   └── app.py              # Main API server
├── notebooks/              # Training notebooks & EDA
└── requirements.txt
```

---

## 🔮 Future Enhancements

- TensorFlow Lite model conversion for edge/mobile inference  
- Vision Transformers (ViT) for spectrogram classification  
- SQL database for long-term user activity profiles  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

### ✔ Recommended Additions

**Generate requirements.txt**

```bash
pip freeze > requirements.txt
```

**Create .gitignore**

```
node_modules/
venv/
__pycache__/
```
````

