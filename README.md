# Heart Stroke Prediction by Sumit 💓

An intelligent, interactive cardiovascular risk screening web application built with **Streamlit** and **Scikit-Learn**. The application assesses heart disease risk using clinical demographics and exercise stress electrocardiogram (ECG) biomarkers, delivering instant probability estimates through an authentic, modern liquid glass interface.

---

## 🌟 Key Features

- **Real-Time Risk Inference**: Instant binary classification (`High Risk` vs. `Low Risk`) coupled with calibrated risk probability estimates via `model.predict_proba()`.
- **Human-Centric Liquid Glass Theme**: Clean, refined aesthetic featuring frosted glass containers, subtle ambient lighting, and high-contrast typography without artificial clutter.
- **Modular Architecture**: Complete separation of concerns with styling managed in an external stylesheet (`assets/style.css`) and clean, Pythonic application code in `app.py`.
- **Structured Clinical Input Layout**: Two-column layout separating routine baseline vitals from exercise stress test indicators.
- **Visual Risk Meter**: Dynamic visual progress indicator displaying risk probability with color transitions.

---

## 📊 Clinical Features & Schema

The model evaluates 11 clinical attributes that are mapped to 15 standardized features:

| Feature | Input Type | Description / Clinical Values |
| :--- | :--- | :--- |
| **Age** | Slider | Patient age in years (`18` – `100`) |
| **Sex** | Selectbox | Biological sex (`Male (M)` / `Female (F)`) |
| **Chest Pain Type** | Selectbox | Angina category: `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `TA` (Typical Angina), `ASY` (Asymptomatic) |
| **Resting BP** | Number Input | Resting blood pressure in mm Hg on admission (`70` – `220`) |
| **Cholesterol** | Number Input | Serum cholesterol in mg/dL (`80` – `600`) |
| **Fasting Blood Sugar** | Selectbox | Fasting blood glucose: `0` (≤ 120 mg/dL), `1` (> 120 mg/dL) |
| **Resting ECG** | Selectbox | Electrocardiogram baseline: `Normal`, `ST` (ST-T wave abnormality), `LVH` (Left ventricular hypertrophy) |
| **Max Heart Rate** | Slider | Peak heart rate achieved during exercise stress test (`60` – `220` bpm) |
| **Exercise Angina** | Selectbox | Angina induced by physical exertion (`N`: No / `Y`: Yes) |
| **Oldpeak** | Slider | ST segment depression induced by exercise relative to rest (`0.0` – `6.0` mm) |
| **ST Slope** | Selectbox | Slope of peak exercise ST segment: `Up` (Upsloping), `Flat` (Flat), `Down` (Downsloping) |

---

## 🏗️ Project Architecture

```
Heart-Stroke-Prediction-by-Sumit/
├── assets/
│   └── style.css                 # Modular stylesheet (liquid glass theme & responsive layout)
├── app.py                        # Main Streamlit application and inference pipeline
├── Columns_heart.pkl             # Serialized list of expected features (15 columns)
├── Logistic_Regression_heart.pkl # Pre-trained Scikit-Learn Logistic Regression model
├── Scaler_heart.pkl              # Fitted StandardScaler for input normalization
├── .gitignore                    # Version control ignore specifications
└── README.md                     # Comprehensive documentation
```

---

## ⚙️ Machine Learning Pipeline

1. **One-Hot Encoding**: Categorical selections (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are encoded into dummy indicators matching the training schema.
2. **Column Alignment**: Missing columns default to `0`, and features are ordered to match `Columns_heart.pkl`.
3. **Feature Scaling**: The feature vector is transformed using `Scaler_heart.pkl` (`StandardScaler`).
4. **Classification & Scoring**:
   - `Logistic_Regression_heart.pkl` computes the binary prediction (`0`: Low Risk, `1`: High Risk).
   - `model.predict_proba()` calculates the risk and baseline health probabilities.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed on your system.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SumitKumar-990/Heart-Stroke-Prediction-by-Sumit.git
   cd Heart-Stroke-Prediction-by-Sumit
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

4. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🩺 Disclaimer

This application is intended strictly for educational, academic, and screening demonstration purposes. It is **not** a diagnostic medical device and does not substitute for professional medical counsel, formal clinical evaluation, or doctor consultation.

---

## 👨‍💻 Author

Developed with ❤️ by **Sumit Kumar**  
GitHub: [@SumitKumar-990](https://github.com/SumitKumar-990)
