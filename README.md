# 🎓 Student Performance Predictor

## 🌐 Live App

👉 https://student-performance-predictor-gnfbl3d8lu6qweac9wyrcx.streamlit.app/

---

## 🚀 Overview

This project predicts a student's exam score based on key academic and lifestyle factors using Machine Learning.

It transforms raw student data into meaningful insights and provides real-time predictions through an interactive web application built with **Streamlit**.

---

## 🧠 Problem Statement

Students often struggle to understand how their daily habits impact academic performance.

This project helps answer:

> *“How do study hours, sleep, attendance, and past performance affect my score?”*

---

## ⚙️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**

---

## 🏗️ Project Structure

```
student-performance-predictor/
│
├── Data/
│   └── student_exam_scores.csv
│
├── Images/
│   ├── Train Output.png
│   └── Test Output.png
│
├── Model/
│   ├── model.ipynb
│   └── model.pkl
│
├── app.py
├── requirements.txt
├── README.md
```

---

## 📊 Model Details

* **Algorithm:** Linear Regression
* **R² Score:** ~85%
* **MAE:** ~2.3 marks
* **RMSE:** ~2.7 marks

👉 The model shows strong generalization and consistent performance on unseen data.

---

## 📈 Features

* 🎯 Real-time score prediction
* 📊 Simple and interactive UI
* 🔒 Input validation for reliable results
* 💡 Smart feedback based on predicted score

---

## 🧪 Sample Prediction

| Study Hours | Sleep | Attendance | Previous Score | Predicted |
| ----------- | ----- | ---------- | -------------- | --------- |
| 6 hrs       | 7 hrs | 85%        | 80             | ~37       |
| 8 hrs       | 7 hrs | 95%        | 90             | ~43       |

---

## 📸 Visualizations

### 📊 Training Data

![Training](Images/Train%20Output.png)

### 📊 Testing Data

![Testing](Images/Test%20Output.png)

---

## ▶️ Run Locally

### 1. Clone repository

```
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run app

```
streamlit run app.py
```

---

## 🔍 Key Insights

* Previous performance has the highest impact on final score
* Study hours and attendance significantly influence results
* Sleep plays a moderate but important role

---

## 🌟 Future Improvements

* 📊 Add interactive graphs and insights
* 🤖 Use advanced models (Random Forest, XGBoost)
* 🎨 Improve UI/UX design
* 📈 Feature importance visualization

---

## 💡 What I Learned

* End-to-end ML workflow (data → model → deployment)
* Model evaluation and performance metrics
* Building and deploying ML apps using Streamlit
* Debugging real-world deployment issues

---

## 🔥 Conclusion

This project demonstrates how machine learning can:

> Transform simple student data into actionable predictions and insights.

---

## 📬 Contact

Feel free to connect and give feedback!
