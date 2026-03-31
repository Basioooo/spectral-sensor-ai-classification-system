# 🌈 AI-Based Spectral Sensor Classification System (AS7265x)

An end-to-end machine learning system for classifying materials and colors using real spectral sensor data from the **AS7265x spectral sensor**.

This project demonstrates applied AI by integrating deep learning, data analysis, and a web-based interface into a complete system ready for real-world usage.

---

## 🚀 Overview

This system uses spectral data (410–940 nm wavelengths) collected from the AS7265x sensor to classify different materials/colors using a **1D Convolutional Neural Network (CNN)**.

The project covers the full machine learning lifecycle:
- Data preprocessing & analysis  
- Model training & evaluation  
- Deployment via a web application  
- Real-time predictions and visualization  

---

## 🧠 Key Features

- 📊 **End-to-End ML Pipeline**
  - Data loading, cleaning, and preprocessing
  - Feature scaling and encoding
  - Model training, validation, and testing  

- 🤖 **Deep Learning Model**
  - 1D CNN architecture for spectral signal processing  
  - Multi-class classification using TensorFlow/Keras  

- 🌐 **Web Application (Flask)**
  - Train/retrain model from UI  
  - Real-time prediction interface  
  - Interactive dashboard for data analysis  

- 📈 **Data Visualization**
  - Class distribution plots  
  - Spectral pattern visualization  
  - Correlation heatmaps  

---

## 🏗️ System Architecture
Sensor Data (AS7265x)
↓
Data Preprocessing (Scaling, Encoding)
↓
1D CNN Model (TensorFlow/Keras)
↓
Evaluation (Accuracy, Classification Report)
↓
Flask Web App
↓
User Input → Real-Time Prediction

---

## 🧪 Technologies Used

- **Programming:** Python  
- **Machine Learning:** TensorFlow, Keras, Scikit-learn  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Flask  

---

## 📂 Project Structure
├── Spec_CNN.py # Main spectral classification system
├── colors.csv # Dataset (colors)
├── materials.csv # Dataset (materials)
├── templates/ # (if separated HTML templates are used)
├── README.md # Project documentation

---

## ⚙️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/yourusername/spectral-sensor-ai-classification-system.git
cd spectral-sensor-ai-classification-system

---

### 2. Install dependencies

pip install -r requirements.txt

---

### 3. Run the application

python Spec_CNN.py

---

### 4. Open in browser
http://127.0.0.1:5000/

---

### 📊 Model Details
Input: 16 spectral wavelength values (AS7265x sensor)
Architecture:
Conv1D layers for feature extraction
Batch normalization & dropout for regularization
Dense layers for classification
Output: Multi-class classification (color/material)

---

### 🔮 Usage
Train the model via /train endpoint
Analyze dataset via /analyze
Input spectral readings to get predictions
View probability distribution for model confidence

---

### 🎯 Applications
Industrial quality control
Material classification
Smart sensor systems
Embedded AI solutions
IoT-based intelligent monitoring

---

### 🤝 Team Contribution

This project was developed as part of a graduation project integrating:

AI & Machine Learning (My Role)
Software Development
Hardware Integration

---

### 📌 Future Improvements
Deploy model to cloud (AWS / Azure)
Integrate real-time sensor streaming
Optimize model performance
Convert to REST API service
Add mobile or IoT integration

---

### 👤 Author

Mahmoud Hesham
Machine Learning Engineer
Data Science & AI Engineering Student @ Elsewedy University of Technology

---

### ⭐ Acknowledgments
AS7265x Spectral Sensor
Open-source ML libraries (TensorFlow, Scikit-learn)

---
