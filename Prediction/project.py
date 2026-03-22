import pickle

# Load model safely
try:
    with open("model/model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception("Model file not found. Check path: model/model.pkl")


def predict_score(hours_studied, sleep_hours, attendance_percent, previous_scores):
    
    # 🔒 Input validation
    if not (0 <= hours_studied <= 24):
        raise ValueError("Study hours must be between 0 and 24")
    
    if not (0 <= sleep_hours <= 24):
        raise ValueError("Sleep hours must be between 0 and 24")
    
    if not (0 <= attendance_percent <= 100):
        raise ValueError("Attendance must be between 0 and 100")
    
    if not (0 <= previous_scores <= 100):
        raise ValueError("Previous score must be between 0 and 100")

    # Prepare input
    input_data = [[hours_studied, sleep_hours, attendance_percent, previous_scores]]

    # Prediction
    prediction = model.predict(input_data)[0]

    return round(prediction, 2)


def get_feedback(score):
    # 🎯 Adjusted for your scale (~0–50)
    if score >= 40:
        return "Excellent performance 🚀"
    elif score >= 30:
        return "Good, but can improve 📈"
    else:
        return "Needs improvement ⚠️"