import streamlit as st
import pickle
import os

import os
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "..", "model", "model.pkl")
        model_path = os.path.abspath(model_path)

        with open(model_path, "rb") as file:
            return pickle.load(file)

    except FileNotFoundError:
        st.error(f"❌ Model not found at: {model_path}")
        return None

model = load_model()
# ==============================
# 🎯 Prediction Function
# ==============================
def predict_score(hours_studied, sleep_hours, attendance_percent, previous_scores):
    
    # Input validation
    if not (0 <= hours_studied <= 24):
        st.error("Study hours must be between 0 and 24")
        return None
    
    if not (0 <= sleep_hours <= 24):
        st.error("Sleep hours must be between 0 and 24")
        return None
    
    if not (0 <= attendance_percent <= 100):
        st.error("Attendance must be between 0 and 100")
        return None
    
    if not (0 <= previous_scores <= 100):
        st.error("Previous score must be between 0 and 100")
        return None

    input_data = [[hours_studied, sleep_hours, attendance_percent, previous_scores]]
    
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# ==============================
# 💬 Feedback Function
# ==============================
def get_feedback(score):
    if score >= 40:
        return "🚀 Excellent performance"
    elif score >= 30:
        return "📈 Good, but can improve"
    else:
        return "⚠️ Needs improvement"

# ==============================
# 🎨 UI Section
# ==============================
st.set_page_config(page_title="Student Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Predict your exam score based on your habits.")

# Inputs
hours = st.number_input("📚 Study Hours", min_value=0.0, max_value=24.0, step=0.5)
sleep = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("📊 Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
previous = st.number_input("📈 Previous Score", min_value=0.0, max_value=100.0, step=1.0)

# Predict Button
if st.button("Predict"):
    if model is not None:
        score = predict_score(hours, sleep, attendance, previous)
        
        if score is not None:
            feedback = get_feedback(score)

            st.success(f"🎯 Predicted Score: {score}")
            st.info(feedback)
    else:
        st.warning("Model is not loaded. Fix the issue and retry.")