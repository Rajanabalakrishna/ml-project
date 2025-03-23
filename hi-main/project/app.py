from flask import Flask, render_template, request, jsonify
import json
import re

app = Flask(__name__)

# Load responses from JSON file
with open('responses.json', 'r') as file:
    data = json.load(file)
    responses = data['responses']

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def process_query():
    user_message = request.json.get('message', '')
    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)
