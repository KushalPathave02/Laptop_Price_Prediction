# 💻 Laptop Price Predictor

A professional Machine Learning web application that predicts laptop prices based on user-provided specifications. This project is built using **Python**, **Flask**, and **Scikit-Learn**, and is ready for deployment on platforms like Render or Heroku.

## 🚀 Live Demo
*(Once deployed, add your Render/Heroku link here)*
Example: [https://laptop-price-predictor-xyz.onrender.com](https://laptop-price-predictor-xyz.onrender.com)

---

## 🛠️ Features
- **Accurate Predictions:** Uses a Random Forest Regressor with ~81% accuracy.
- **Responsive UI:** Clean and modern interface built with custom CSS.
- **Pre-processing Pipeline:** Automated data cleaning and feature engineering.
- **API Support:** Includes a JSON API endpoint for integration with other apps.
- **Production Ready:** Configured with `Gunicorn`, `Procfile`, and `runtime.txt` for easy deployment.

---

## 📊 Dataset
The model is trained on a comprehensive laptop dataset containing:
- Company (Brand)
- Type (Notebook, Gaming, Ultrabook, etc.)
- RAM (GB)
- Weight (kg)
- Screen Resolution (IPS, Touchscreen)
- CPU & GPU Brands
- Operating System

---

## 🧠 Model Information
- **Algorithm:** Random Forest Regressor (300 estimators)
- **Target Variable:** Price (Log-transformed for better normality)
- **Accuracy (R2 Score):** ~0.81
- **Preprocessing:** One-Hot Encoding for categorical variables, Numeric conversion for RAM/Weight.

---

## 📁 Project Structure
```text
LaptopPricePrediction/
│
├── data/
│   └── laptop_data.csv      # Raw dataset
├── model/
│   ├── laptop_model.pkl     # Trained ML Pipeline
│   └── df.pkl               # Processed dataframe for UI dropdowns
├── static/
│   └── style.css            # Custom styling
├── templates/
│   └── index.html           # Main UI template
├── app.py                   # Flask Backend & API
├── preprocess.py            # Data cleaning & Model training script
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment instructions
└── runtime.txt              # Python version for deployment
```

---

## ⚙️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KushalPathave02/Laptop_Price_Prediction.git
   cd Laptop_Price_Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional - files are already included):**
   ```bash
   python3 preprocess.py
   ```

4. **Run the app:**
   ```bash
   python3 app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

---

## 🌐 API Endpoint
- **URL:** `/api/predict`
- **Method:** `POST`
- **Payload Example:**
  ```json
  {
    "company": "Dell",
    "type": "Notebook",
    "ram": 8,
    "weight": 1.9,
    "touchscreen": 0,
    "ips": 1,
    "cpu": "Intel",
    "gpu": "Nvidia",
    "os": "Windows 10"
  }
  ```

---

## 🚀 Deployment (Render.com)
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `gunicorn app:app`

---

## 👤 Author
**Kushal Pathave**
- GitHub: [@KushalPathave02](https://github.com/KushalPathave02)
