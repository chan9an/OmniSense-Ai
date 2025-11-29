````markdown
# 🩺 OmniSense AI: Multi-Modal Activity & Cough Detection System

## Overview

OmniSense AI is a comprehensive, end-to-end platform that leverages machine learning and sensor fusion to provide real-time human activity recognition (HAR) and respiratory event monitoring (cough detection).

The project successfully integrates deep learning models for complex time-series analysis with a responsive web application, demonstrating expertise across the entire **MLOps** and **Full-Stack** development lifecycle.

---

## 🛠️ Technical Stack & Tools

| Component | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Data Collection** | Phyphox (Mobile App) | Real-time streaming of raw accelerometer and gyroscope data. |
| **Backend/API** | **Python**, **Flask** | Model inference, data ingestion, and exposing prediction endpoints. |
| **Frontend/Visualization** | **React** | Interactive dashboard for real-time activity and cough status display. |
| **Machine Learning** | **Keras (CNN)**, **TensorFlow**, **scikit-learn** | Building, training, and evaluating deep learning and classical classification models. |
| **Data Science** | NumPy, Pandas | Sensor data preprocessing, feature engineering, and pipeline management. |

---

## ✨ Key Features & Technical Depth

* **Multi-Modal Data Processing:** Architected a single platform to handle two distinct data types: **time-series** sensor data (for HAR) and **audio** data (for cough detection).
* **Deep Learning (DL) Pipeline:** Developed a **Convolutional Neural Network (CNN)** using Keras for robust feature extraction and classification of complex time-series sensor patterns.
* **Model Benchmarking:** Employed analytical rigor by benchmarking the DL model against classical machine learning algorithms (Random Forest, SVM) using **scikit-learn** to validate superior accuracy.
* **Real-Time Deployment:** Deployed the trained models using a **Python (Flask) REST API**, allowing the external React frontend to request and display activity predictions with low latency.
* **End-to-End System:** Built a complete full-stack system from sensor data ingestion (Phyphox streaming) to UI visualization (React).

---

## 🚀 Future Scope & Improvements

The modular architecture of OmniSense AI allows for easy expansion:

1.  **Direct Memory Reading (Advanced):** Replace log/stream parsing with advanced techniques (e.g., memory tracing) for zero-latency event detection.
2.  **Model Optimization:** Explore using **PyTorch** for model training and quantization for deployment on edge devices.
3.  **UI/UX:** Develop a dedicated mobile application (instead of using Phyphox) for seamless, always-on sensor data capture and user interaction.

---

## ⚙️ Getting Started

To run this project locally, you will need Python 3.9+ and Node.js (for the React frontend).

### Prerequisites

```bash
# 1. Clone the repository
git clone [https://github.com/chan9an/OmniSense-AI.git](https://github.com/chan9an/OmniSense-AI.git)
cd OmniSense-AI
````

### Backend Setup (Python/Flask)

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the required packages:
    ```bash
    pip install Flask scikit-learn tensorflow pandas numpy
    ```
3.  Run the Flask API (ensuring network accessibility for sensor streaming):
    ```bash
    python main_server.py
    ```
    *The server will run on `http://0.0.0.0:5000`.*

### Frontend Setup (React)

*(Assuming frontend code lives in a subdirectory named `client/`)*

1.  Navigate to the frontend directory:
    ```bash
    cd client
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the React application:
    ```bash
    npm start
    ```

The application will open in your browser, connecting in real-time to the Flask API for prediction updates.

```
```
