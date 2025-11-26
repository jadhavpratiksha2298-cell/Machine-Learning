import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load model & scaler
# -----------------------------
MODEL_PATH = r"C:\Users\admīn\Desktop\pratiksha\ML project\iris project\log_reg_model.pkl"
SCALER_PATH = r"C:\Users\admīn\Desktop\pratiksha\ML project\iris project\scaler.pkl"

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
CLASS_MAPPING = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# -----------------------------
# Custom CSS for Aesthetic UI
# -----------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Gradient Title */
.title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

/* Card style */
.card {
    background: #ffffffaa;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    padding: 18px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    color: #000;
}

/* Button style */
.stButton>button {
    background: linear-gradient(90deg,#2575fc,#6a11cb);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 class='title'>🌸 Iris Flower Prediction</h1>", unsafe_allow_html=True)
st.write("Beautiful UI • Logistic Regression • Scaled Inputs")

# -----------------------------
# Manual Single Prediction
# -----------------------------
st.subheader("✨ Predict Single Sample")

col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)

with col2:
    pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    data = np.array([[sl, sw, pl, pw]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0]

    st.markdown(f"""
    <div class='card'>
        <div class='pred-box'>
            🌼 Prediction: {CLASS_MAPPING[pred]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("🔢 Probabilities:")
    st.write({CLASS_MAPPING[i]: float(proba[i]) for i in range(3)})