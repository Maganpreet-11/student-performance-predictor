import streamlit as st
from project import predict_score, get_feedback

st.title("🎓 Student Performance Predictor")

# User Inputs
hours = st.number_input("Study Hours", min_value=0.0)
sleep = st.number_input("Sleep Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous = st.number_input("Previous Score", min_value=0.0)

# Button
if st.button("Predict"):
    score = predict_score(hours, sleep, attendance, previous)
    feedback = get_feedback(score)

    st.success(f"Predicted Score: {score}")
    st.info(feedback)