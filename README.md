# 🩺 OmniSense AI — Multi-Modal Activity & Health Monitoring

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-green)
![React](https://img.shields.io/badge/Frontend-React-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📘 Overview

OmniSense AI is an end-to-end multi-modal intelligence system that performs real-time **Human Activity Recognition (HAR)** and **respiratory event detection**.

The platform fuses motion-sensor time-series data with audio-based spectral features to identify physical activities and detect events such as coughing or sneezing. It includes everything from data preprocessing and model training to a real-time backend and a polished React dashboard.

---

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Deep Learning** | :contentReference[oaicite:0]{index=0} (CNN), :contentReference[oaicite:1]{index=1} |
| **Machine Learning** | :contentReference[oaicite:2]{index=2} (Random Forest), NumPy, Pandas |
| **Backend API** | Python with :contentReference[oaicite:3]{index=3} |
| **Frontend** | JavaScript with :contentReference[oaicite:4]{index=4} |
| **Signal & Audio Processing** | Librosa, SciPy |

---

## ✨ Key Features

### 🔹 Multi-Modal Deep Learning  
Processes two coordinated data streams:
- **Motion Sensors:** 1D-CNN models classify activities such as walking, running, or standing.  
- **Audio:** Spectrogram-based models detect respiratory cues like coughs or sneezes.

### 🔹 Benchmarking  
Includes a comparison between the custom CNN architecture and a classical Random Forest baseline.

### 🔹 Real-Time Performance  
A lightweight Flask API supports **sub-200ms inference**, enabling instant response on the client dashboard.

### 🔹 Interactive Web Dashboard  
A responsive React UI visualizes:
- Live activity predictions  
- Real-time audio-based health events  
- System status and inference metrics  

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
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend API
python app.py
```

Backend runs at: **http://localhost:5000**

---

## 3. Frontend Setup (React)

```bash
cd client
npm install
npm start
```

Dashboard runs at: **http://localhost:3000**

---

## 📁 Project Structure

```text
OmniSense-AI/
├── client/                 # React frontend
│   ├── src/components/     # UI components & visualizations
│   └── public/
├── server/                 # Flask backend
│   ├── models/             # Trained .h5 / .pkl models
│   ├── processing/         # Signal + audio processing scripts
│   └── app.py              # API entry point
├── notebooks/              # Training, EDA, experiments
└── requirements.txt
```

---

## 🔮 Future Enhancements

- **On-Device Edge Deployment:** Convert models to TensorFlow Lite for mobile/embedded inference.  
- **Transformer Models:** Explore Vision Transformers for improved spectrogram classification.  
- **User Profiles & History:** Integrate a SQL database for long-term activity and event trends.

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open an issue or submit a pull request for new features, bug fixes, or enhancements.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

### ✔ Recommended Additions
- Generate `requirements.txt`:
  ```bash
  pip freeze > requirements.txt
  ```
- Add a `.gitignore` including:
  ```
  node_modules/
  venv/
  __pycache__/
  ```
