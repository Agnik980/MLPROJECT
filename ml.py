import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_excel(r"C:/Users/CSE-M77/Desktop/htmlproject/Heart-D-Dredesigned (1).xlsx")

# 2. Features and target split
y = df["restecg"]
X = df.drop(["restecg"], axis=1)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {acc:.2f}%")

# ==========================================
# 6. Save model to a Pickle file
# ==========================================

# Option A: Using standard pickle
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Option B: Using joblib (more efficient for NumPy/Scikit-learn models)
joblib.dump(model, "heart_disease_model_joblib.pkl")

print("Model successfully saved as a .pkl file!")