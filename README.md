````markdown
# 🩺 OmniSense AI: Multi-Modal Activity & Health Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-green)
![React](https://img.shields.io/badge/Frontend-React-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

**OmniSense AI** is a robust, multi-modal AI platform designed for real-time **Human Activity Recognition (HAR)** and **respiratory event detection**.

By fusing time-series sensor data (accelerometer/gyroscope) with audio input, the system can accurately classify physical activities and detect health events like coughing or sneezing. This project demonstrates a complete end-to-end pipeline—from raw data ingestion to a user-friendly React dashboard.

---

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Deep Learning** | **Keras (CNN)**, TensorFlow |
| **Machine Learning** | **scikit-learn** (Random Forest Baseline), NumPy, Pandas |
| **Backend API** | **Python (Flask)**, RESTful Architecture |
| **Frontend** | **React**, JavaScript, CSS Modules |
| **Data Processing** | Librosa (Audio), SciPy (Signal Processing) |

---

## ✨ Key Features

* **Multi-Modal Architecture:** Simultaneously processes two distinct data streams:
    * **Motion:** 1D-CNN models analyze time-series data to detect activities (Walking, Standing, Running).
    * **Audio:** Spectral analysis detects specific respiratory sounds (Coughs, Sneezes).
* **Deep Learning vs. Classical ML:** Includes a comparative study benchmarking the custom **Convolutional Neural Network (CNN)** against Random Forest classifiers to validate performance gains.
* **Real-Time Inference:** Engineered a low-latency **Flask** backend that serves predictions to the client in under 200ms.
* **Interactive Dashboard:** A responsive **React** frontend that visualizes live activity status and health alerts.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* Node.js & npm

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/OmniSense-AI.git](https://github.com/yourusername/OmniSense-AI.git)
cd OmniSense-AI
````

### 2\. Backend Setup (Flask)

```bash
cd server
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py
```

*Server running at `http://localhost:5000`*

### 3\. Frontend Setup (React)

```bash
cd client
# Install dependencies
npm install

# Start the dashboard
npm start
```

*Client running at `http://localhost:3000`*

-----

## 📂 Project Structure

```text
OmniSense-AI/
├── client/                 # React Frontend logic
│   ├── src/components/     # Visualization components
│   └── public/
├── server/                 # Python Backend
│   ├── models/             # Saved .h5 and .pkl models
│   ├── processing/         # Feature extraction scripts
│   └── app.py              # Flask API entry point
├── notebooks/              # Jupyter notebooks for training & EDA
└── requirements.txt
```

## 🔮 Future Improvements

  * **Edge Deployment:** Quantize models using TensorFlow Lite for on-device inference (Android/iOS).
  * **Transformer Integration:** Experiment with Vision Transformers (ViT) for spectrogram classification to improve audio accuracy.
  * **User Profiles:** Add database integration (SQL) to track historical user activity trends over time.

-----

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

````

### **Don't forget to:**
1.  **Create a `requirements.txt`** file in your server folder if you haven't already:
    ```bash
    pip freeze > requirements.txt
    ```
2.  **Add a `.gitignore`** to exclude `node_modules`, `venv`, and `__pycache__`.
````
