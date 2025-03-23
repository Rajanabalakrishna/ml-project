import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create the static directory if it doesn't exist
static_dir = os.path.join("C:\\New folder (5)\\hi-main\\project\\static")
os.makedirs(static_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("C:\\New folder (5)\\hi-main\\project\\dataset\\hypothyroid.csv")

# Preprocess data
label_encoders = {}
for col in ['sex', 'on thyroxine', 'sick', 'pregnant', 'goitre', 'tumor', 'output']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset
X = df.drop(columns=['output'])
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)

# Find the best model
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]

# Save best model
import joblib
model_path = os.path.join("C:\\New folder (5)\\hi-main\\project", "best_model.pkl")
joblib.dump(best_model, model_path)

# Plot accuracy
plt.figure(figsize=(8, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()  # Ensure text fits in the figure

# Save the plot with full path
plot_path = os.path.join(static_dir, "accuracy_plot.png")
plt.savefig(plot_path)

print(f"Best model: {best_model_name} with accuracy: {accuracy_scores[best_model_name]:.4f}")
print(f"Model saved to: {model_path}")
print(f"Plot saved to: {plot_path}")
