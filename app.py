from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_path = r"C:\Users\danie\PycharmProjects\FineTuneBert\saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()  # Set model to evaluation mode

# Mapping dictionary
URGENCY_MAP = {0: "Low", 1: "Medium", 2: "High"}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get data from request
    complaint_text = data.get("complaint")

    if not complaint_text:
        return jsonify({"error": "No complaint text provided"}), 400

    # Tokenize input
    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    urgency = URGENCY_MAP.get(predicted_class, "Unknown")

    return jsonify({"complaint": complaint_text, "urgency": urgency})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
