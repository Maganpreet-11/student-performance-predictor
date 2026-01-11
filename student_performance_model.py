import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Data.csv")   # make sure file is in same folder

print("First 5 rows of dataset:")
print(df.head())

# -----------------------------
# 2. Separate Features & Target
# -----------------------------
X = df[['Hours']]     # Independent variable
y = df['Scores']     # Dependent variable

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# -----------------------------
# 6. Model Parameters
# -----------------------------
print("\nModel Parameters:")
print("Slope (Hours → Marks):", model.coef_[0])
print("Intercept:", model.intercept_)

# -----------------------------
# 7. Model Evaluation
# -----------------------------
print("\nModel Performance:")
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))

print("\nTest R²:", r2_score(y_test, y_test_pred))

# -----------------------------
# 8. Predictions for New Values
# -----------------------------
hours_8 = model.predict([[8]])[0]
hours_10 = model.predict([[10]])[0]

print("\nPredictions:")
print("Marks for 8 hours:", hours_8)
print("Marks for 10 hours:", hours_10)

# -----------------------------
# 9. Visualization
# -----------------------------
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, model.predict(X), color='black', linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.grid()
plt.show()
