from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained NLP model from the local path (or from Hugging Face if not local)
nlp_model = pipeline("sentiment-analysis", model="./mental_health_model")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    # Get the user input from the form
    user_input = request.form.get("message")  # Use .get() to safely access 'message'
    
    if not user_input:
        return jsonify({"response": "Please provide a valid message."})

    response = generate_response(user_input)
    return jsonify({"response": response})

def generate_response(user_input):
    # Use NLP model to analyze the input
    sentiment = nlp_model(user_input)[0]
    
    if sentiment['label'] == 'POSITIVE':
        return "I'm glad to hear that! How can I assist you further?"
    else:
        return "I'm here for you. It's okay to feel this way. Would you like to talk about it?"

if __name__ == "__main__":
    app.run(debug=True)
