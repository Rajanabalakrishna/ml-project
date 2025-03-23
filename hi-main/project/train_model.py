import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("C:\\New folder (5)\\hi-main\\project\\dataset\\hypothyroid.csv")

# Encode categorical features
le = LabelEncoder()
for col in ["sex", "on thyroxine", "sick", "pregnant", "goitre", "tumor", "output"]:
    df[col] = le.fit_transform(df[col])

# Split dataset
X = df.drop(columns=["output"])  # Features
y = df["output"]  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

results = []  # Store model results

# Train & Evaluate Each Model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, f"models/{name}.pkl")

    # Store results
    results.append({"model": name, "accuracy": accuracy, "precision": precision, "f1_score": f1})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results as CSV for Flask to use
results_df.to_csv("model_results.csv", index=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(results_df["model"], results_df["accuracy"], color=["blue", "green", "red", "purple"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("static/accuracy_chart.png")  # Save image for Flask
plt.close()
