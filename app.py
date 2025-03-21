import os
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import subprocess

# Ensure LFS files are pulled
subprocess.run(["git", "lfs", "pull"], check=True)


# Load model path from environment variable (or default to "saved_model")
model_path = os.getenv("MODEL_PATH", "saved_model")

# Load model and tokenizer
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
    print(f"Loading model from: {model_path}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

