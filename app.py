import os
import pickle
import streamlit as st

# ==============================
# 🔁 Load Model (Silent + Safe)
# ==============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)

    possible_paths = [
        os.path.join(base_dir, "model.pkl"),
        os.path.join(base_dir, "..", "model.pkl"),
        os.path.join(base_dir, "..", "model", "model.pkl"),
    ]

    for path in possible_paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            with open(path, "rb") as file:
                return pickle.load(file)

    return None  # silent fail

model = load_model()

# ==============================
# 🎯 Prediction Function
# ==============================
def predict_score(hours_studied, sleep_hours, attendance_percent, previous_scores):
    
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
# 💬 Feedback
# ==============================
def get_feedback(score):
    if score >= 40:
        return "🚀 Excellent performance"
    elif score >= 30:
        return "📈 Good, but can improve"
    else:
        return "⚠️ Needs improvement"

# ==============================
# 🎨 UI
# ==============================
st.set_page_config(page_title="Student Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Predict your exam score based on your habits.")

hours = st.number_input("📚 Study Hours", 0.0, 24.0, step=0.5)
sleep = st.number_input("😴 Sleep Hours", 0.0, 24.0, step=0.5)
attendance = st.number_input("📊 Attendance (%)", 0.0, 100.0, step=1.0)
previous = st.number_input("📈 Previous Score", 0.0, 100.0, step=1.0)

if st.button("Predict"):
    if model is None:
        st.error("Model not found. Please check deployment.")
    else:
        score = predict_score(hours, sleep, attendance, previous)
        if score is not None:
            st.success(f"🎯 Predicted Score: {score}")
            st.info(get_feedback(score))
