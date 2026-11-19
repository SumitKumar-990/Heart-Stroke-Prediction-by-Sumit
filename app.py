import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Setup & External Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Stroke Prediction by Sumit💓",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css(file_path: str = "assets/style.css"):
    """Load external CSS stylesheet for clean separation of concerns."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

load_css()

# ---------------------------------------------------------
# Load Pretrained ML Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Logistic_Regression_heart.pkl")
    scaler = joblib.load("Scaler_heart.pkl")
    columns = joblib.load("Columns_heart.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.html("""
<div class="header-box">
    <h1 class="header-title">Heart Stroke Prediction by Sumit💓</h1>
    <p class="header-subtitle">Provide the following details</p>
</div>
""")

# ---------------------------------------------------------
# Clinical Input Form (2 Columns)
# ---------------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.html("""
    <div class="section-title">
        <span>👤 Demographics & Baseline Vitals</span>
    </div>
    """)

    age = st.slider("Age", min_value=18, max_value=100, value=40, step=1,
                    help="Patient age in years.")

    sex_display = st.selectbox(
        "Sex",
        options=["M — Male", "F — Female"],
        index=0,
        help="Biological sex of the patient."
    )
    sex = "M" if sex_display.startswith("M") else "F"

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=70, max_value=220, value=120, step=1,
        help="Resting blood pressure in mm Hg."
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=80, max_value=600, value=200, step=1,
        help="Serum cholesterol in mg/dL."
    )

    fasting_bs_display = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[
            "0 — Normal (≤ 120 mg/dL)",
            "1 — Elevated (> 120 mg/dL)"
        ],
        index=0,
        help="Fasting blood sugar level threshold."
    )
    fasting_bs = 1 if fasting_bs_display.startswith("1") else 0

with col_right:
    st.html("""
    <div class="section-title">
        <span>⚡ Cardiac Stress & ECG Biomarkers</span>
    </div>
    """)

    chest_pain_display = st.selectbox(
        "Chest Pain Type",
        options=[
            "ATA — Atypical Angina",
            "NAP — Non-Anginal Pain",
            "TA — Typical Angina",
            "ASY — Asymptomatic"
        ],
        index=0,
        help="Classification of chest pain."
    )
    chest_pain = chest_pain_display.split(" — ")[0]

    resting_ecg_display = st.selectbox(
        "Resting ECG",
        options=[
            "Normal — Normal",
            "ST — ST-T Wave Abnormality",
            "LVH — Left Ventricular Hypertrophy"
        ],
        index=0,
        help="Resting electrocardiogram results."
    )
    resting_ecg = resting_ecg_display.split(" — ")[0]

    max_hr = st.slider(
        "Max Heart Rate",
        min_value=60, max_value=220, value=150, step=1,
        help="Maximum heart rate achieved during exercise stress test."
    )

    exercise_angina_display = st.selectbox(
        "Exercise-Induced Angina",
        options=[
            "N — No",
            "Y — Yes"
        ],
        index=0,
        help="Whether physical exercise induces chest angina."
    )
    exercise_angina = "Y" if exercise_angina_display.startswith("Y") else "N"

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        min_value=0.0, max_value=6.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest."
    )

    st_slope_display = st.selectbox(
        "ST Slope",
        options=[
            "Up — Upsloping",
            "Flat — Flat",
            "Down — Downsloping"
        ],
        index=0,
        help="Peak exercise ST segment slope."
    )
    st_slope = st_slope_display.split(" — ")[0]

st.html("<div style='height: 10px;'></div>")

# ---------------------------------------------------------
# Prediction Execution & Results
# ---------------------------------------------------------
if st.button("Predict", use_container_width=True):
    # Construct raw feature map matching training encoding
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Impute missing dummy columns with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure strictly correct column ordering
    input_df = input_df[expected_columns]

    # Normalize inputs via fitted scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction and calculate calibrated probabilities
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    risk_proba = float(probabilities[1]) * 100.0
    safe_proba = float(probabilities[0]) * 100.0

    if prediction == 1:
        st.html(f"""<div class="result-card-danger">
    <div class="result-header">
        <div>
            <h3 class="result-title">⚠️ High Risk of Heart Disease</h3>
        </div>
        <div style="text-align: right;">
            <div class="result-score-val" style="color: #fb7185;">{risk_proba:.1f}%</div>
            <div class="result-score-lbl">Estimated Risk</div>
        </div>
    </div>
    <div class="risk-meter">
        <div class="risk-meter-fill" style="width: {min(max(risk_proba, 4.0), 100.0):.1f}%; background: #f43f5e;"></div>
    </div>
    <div class="stat-grid">
        <div class="stat-pill">
            <div class="stat-val" style="color: #fb7185;">High Risk</div>
            <div class="stat-lbl">Classification</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">{safe_proba:.1f}%</div>
            <div class="stat-lbl">Health Probability</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">Logistic Regression</div>
            <div class="stat-lbl">Model</div>
        </div>
    </div>
    <div class="result-notes">
        <strong>Clinical Note:</strong> The model identified an elevated likelihood of cardiovascular risk based on the provided inputs. Please consider consulting a doctor or cardiologist for diagnostic testing and personalized advice.
    </div>
</div>""")
    else:
        st.html(f"""<div class="result-card-safe">
    <div class="result-header">
        <div>
            <h3 class="result-title">✅ Low Risk of Heart Disease</h3>
        </div>
        <div style="text-align: right;">
            <div class="result-score-val" style="color: #34d399;">{risk_proba:.1f}%</div>
            <div class="result-score-lbl">Estimated Risk</div>
        </div>
    </div>
    <div class="risk-meter">
        <div class="risk-meter-fill" style="width: {min(max(risk_proba, 4.0), 100.0):.1f}%; background: #10b981;"></div>
    </div>
    <div class="stat-grid">
        <div class="stat-pill">
            <div class="stat-val" style="color: #34d399;">Low Risk</div>
            <div class="stat-lbl">Classification</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">{safe_proba:.1f}%</div>
            <div class="stat-lbl">Health Probability</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">Logistic Regression</div>
            <div class="stat-lbl">Model</div>
        </div>
    </div>
    <div class="result-notes">
        <strong>Clinical Note:</strong> Current indicators indicate a healthy cardiovascular risk profile. Continue maintaining balanced nutrition, regular exercise, and standard routine wellness checkups.
    </div>
</div>""")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.html("""
<div class="footer-text">
    Heart Stroke Prediction • Developed with ❤️ by Sumit
    <div class="footer-disclaimer">
        Disclaimer: This tool is intended for educational and screening demonstration. It does not replace professional clinical diagnosis or medical counsel.
    </div>
</div>
""")