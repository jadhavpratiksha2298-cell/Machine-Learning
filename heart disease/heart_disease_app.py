import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model and Scaler
# ----------------------------
model = pickle.load(open("Heart_disease_model.pickle", "rb"))
scaler = pickle.load(open("scaler (1).pkl", "rb"))

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .title { text-align: center; color:#c0392b; font-size:40px; font-weight:700; }
    .subtitle { text-align: center; color:#2c3e50; font-size:18px; }
    .stButton>button {
        background-color:#c0392b;
        color:white;
        padding:0.6rem 1.2rem;
        border-radius:10px;
        border:none;
        font-size:18px;
    }
    .stButton>button:hover {
        background-color:#922b21;
        color:white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.markdown("<h1 class='title'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter the details below to check your heart disease risk.</p>", unsafe_allow_html=True)
st.write("")

# ----------------------------
# Input Form
# ----------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
        restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250)
        exang = st.selectbox("Exercise-induced Angina (1 = Yes, 0 = No)", [0, 1])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)

    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible)", [1, 2, 3])

    submitted = st.form_submit_button("Predict ❤️")

# ----------------------------
# Prediction Logic
# ----------------------------

if submitted:
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]

    st.write("---")
    if prediction == 1:
        st.error("⚠️ High chance of Heart Disease! Please consult a doctor.")
    else:
        st.success("✅ No significant risk of heart disease detected.")
