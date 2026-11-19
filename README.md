# Heart Stroke Prediction by Sumit 💓

An intelligent, interactive cardiovascular risk screening web application built with **Streamlit** and **Scikit-Learn**. The application assesses heart disease risk using clinical demographics and exercise stress electrocardiogram (ECG) biomarkers, delivering instant probability estimates through an authentic, modern liquid glass interface.

---

## 🌟 Key Features

- **Real-Time Risk Inference**: Instant binary classification (`High Risk` vs. `Low Risk`) coupled with calibrated risk probability estimates via `model.predict_proba()`.
- **Benchmarked ML Architecture**: Based on rigorous model comparison across 5 classification algorithms in `Heart.ipynb`, selecting the top-performing model (**86.96% Accuracy**, **0.8846 F1-Score**).
- **Human-Centric Liquid Glass Theme**: Clean, refined aesthetic featuring frosted glass containers, subtle ambient lighting, and high-contrast typography without artificial clutter.
- **Modular Architecture**: Complete separation of concerns with styling managed in an external stylesheet (`assets/style.css`) and clean, Pythonic application code in `app.py`.
- **Structured Clinical Input Layout**: Two-column layout separating routine baseline vitals from exercise stress test indicators.
- **Visual Risk Meter**: Dynamic visual progress indicator displaying risk probability with color transitions.

---

## 🧪 Model Training, Comparison & Selection (`Heart.ipynb`)

In the [Heart.ipynb](Heart.ipynb) Jupyter Notebook, I took the cardiovascular dataset, conducted exploratory data analysis (EDA), applied one-hot encoding to categorical features, and normalized the numerical features using `StandardScaler`.

To ensure optimal predictive performance, I trained and evaluated **5 different machine learning classifiers** under an identical 80/20 train-test split:

| Model | Accuracy | F1-Score | Status |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **86.96%** | **0.8846** | 🏆 **Selected Best Model & Deployed** |
| **K-Nearest Neighbors (KNN)** | 86.41% | 0.8815 | Evaluated |
| **Support Vector Classifier (SVM)** | 84.78% | 0.8667 | Evaluated |
| **Naive Bayes (GaussianNB)** | 84.78% | 0.8614 | Evaluated |
| **Decision Tree** | 80.43% | 0.8200 | Evaluated |

### Why Logistic Regression Was Chosen:
1. **Top Performance**: Achieved the highest accuracy (**86.96%**) and highest F1-score (**0.8846**) among all benchmarked algorithms.
2. **Probability Calibration**: Naturally outputs reliable, well-calibrated posterior probabilities (`predict_proba`), enabling the interactive risk meter in the web UI.
3. **Robust Generalization**: Resists overfitting on clinical tabular data while remaining computationally efficient and highly interpretable.

The winning model, along with the fitted scaler and feature schema, was serialized with `joblib` into `Logistic_Regression_heart.pkl`, `Scaler_heart.pkl`, and `Columns_heart.pkl` for production inference.

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
├── Heart.ipynb                   # Jupyter notebook (EDA, preprocessing, training & model comparison)
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
   pip install streamlit pandas scikit-learn joblib jupyter
   ```

4. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```

   To explore the model training and comparison notebook:
   ```bash
   jupyter notebook Heart.ipynb
   ```

---

## 🩺 Disclaimer

This application is intended strictly for educational, academic, and screening demonstration purposes. It is **not** a diagnostic medical device and does not substitute for professional medical counsel, formal clinical evaluation, or doctor consultation.

---

## 👨‍💻 Author

Developed with ❤️ by **Sumit Kumar**  
GitHub: [@SumitKumar-990](https://github.com/SumitKumar-990)
