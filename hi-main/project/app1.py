from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import re
import os

app = Flask(__name__, static_folder="static")

# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Try to load model results
try:
    results_df = pd.read_csv("model_results.csv")
    # Get the best model (highest F1-score)
    best_model = results_df.loc[results_df["f1_score"].idxmax(), "model"]
except (FileNotFoundError, KeyError):
    # Default values if file not found
    results_df = pd.DataFrame({"model": ["Default"], "accuracy": [0], "precision": [0], "f1_score": [0]})
    best_model = "Default"

# Load chatbot responses
try:
    with open('responses.json', 'r') as file:
        data = json.load(file)
        responses = data['responses']
except FileNotFoundError:
    # Default responses if file not found
    responses = {
        "greetings": "Hello! How can I assist you with thyroid prediction today?",
    "what is thyroid": "The thyroid is a butterfly-shaped gland in the neck that produces hormones regulating metabolism, energy, and overall body functions.",
    "what are symptoms of thyroid": "Symptoms vary but may include fatigue, weight changes, depression, hair thinning, irregular heartbeat, and difficulty tolerating cold or heat.",
    "how does this model work": "Our model processes your inputs using machine learning algorithms to predict whether you may have thyroid disease.",
    "which algorithm is used": "We use Logistic Regression, Random Forest, SVM, and KNN to train the model and determine the best-performing one.",
    "what precautions should be taken": "Precautions include maintaining a healthy diet, managing stress, avoiding excessive iodine intake, and consulting a doctor regularly.",
    "what are the thyroid levels": "Thyroid levels are usually measured through TSH, T3, and T4 hormone levels. The common categories are:\n- **Normal:** TSH (0.4 - 4.0 mIU/L)\n- **Hypothyroidism (Underactive Thyroid):** TSH > 4.5 mIU/L\n- **Hyperthyroidism (Overactive Thyroid):** TSH < 0.4 mIU/L",
    "what is hypothyroidism": "Hypothyroidism occurs when the thyroid gland does not produce enough hormones, leading to symptoms like fatigue, weight gain, and cold sensitivity.",
    "what is hyperthyroidism": "Hyperthyroidism happens when the thyroid gland produces too many hormones, causing symptoms like rapid heartbeat, weight loss, and nervousness.",
    "positive": "Based on your inputs, there is a possibility of thyroid disorder. Please consult a doctor for further analysis.",
    "negative": "Your inputs suggest no thyroid disorder. However, consulting a medical professional for a full diagnosis is always a good idea.",
    "precautions hypothyroidism": "For hypothyroidism, maintain a balanced diet rich in iodine, selenium, and zinc. Take prescribed medications regularly and avoid excessive soy intake.",
    "precautions hyperthyroidism": "For hyperthyroidism, avoid excessive iodine intake, manage stress, and take prescribed medication as per your doctor’s advice.",
    "thyroid level normal": "Your thyroid hormone levels are within the normal range.",
    "thyroid level hypothyroid": "Your thyroid hormone levels indicate hypothyroidism. It's recommended to consult a doctor.",
    "thyroid level hyperthyroid": "Your thyroid hormone levels suggest hyperthyroidism. A medical consultation is advised.",
    "fallback": "I'm sorry, I didn't understand that. Can you please rephrase your question?",
    "bye": "Goodbye! Stay healthy and take care!"
    }

def get_response(user_input):
    # Convert input to lowercase and remove punctuation
    processed_input = re.sub(r'[^\w\s]', '', user_input.lower())
    
    # Check for exact matches
    if processed_input in responses:
        return responses[processed_input]
    
    # Check for partial matches
    for key in responses:
        if key in processed_input:
            return responses[key]
    
    # Default response if no match found
    return "I'm sorry, I don't have an answer for that."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = request.form["age"]
            sex = request.form["sex"]
            on_thyroxine = request.form["on_thyroxine"]
            sick = request.form["sick"]
            pregnant = request.form["pregnant"]
            goitre = request.form["goitre"]
            tumor = request.form["tumor"]

            # Convert inputs for the model
            input_data = np.array([[int(age), int(sex == "female"), int(on_thyroxine == "t"),
                                   int(sick == "t"), int(pregnant == "t"), int(goitre == "t"), int(tumor == "t")]])
            
            # Load the best model - adjust path as needed
            try:
                model = joblib.load(f"models/{best_model}.pkl")
                # Make a prediction
                prediction = model.predict(input_data)[0]
                result = "Positive" if prediction == 1 else "Negative"
            except:
                result = "Error loading model"
            
            return render_template("result.html", prediction=result, 
                                  results=results_df.to_dict(orient="records"), 
                                  best_model=best_model)
        except Exception as e:
            return f"Error processing form: {str(e)}"

    return render_template("index.html")

@app.route('/get_response', methods=['POST'])
def process_query():
    try:
        user_message = request.json.get('message', '')
        bot_response = get_response(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Create a placeholder image if it doesn't exist
def create_placeholder_image():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        models = ["Random Forest", "Logistic Regression", "SVM", "Naive Bayes"]
        accuracies = np.random.rand(4) * 0.5 + 0.5  # Random accuracies between 0.5 and 1.0
        
        plt.bar(models, accuracies)
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        
        # Save with both possible names
        plt.savefig("static/accuracy_plot.png")
        plt.savefig("static/accuracy_chart.png")
        plt.close()
    except:
        pass  # Silently fail if matplotlib is not available

# Create the image before running the app
create_placeholder_image()

if __name__ == "__main__":
    app.run(debug=True)
