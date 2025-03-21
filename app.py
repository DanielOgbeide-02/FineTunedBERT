import os
import gdown
import zipfile
import torch
from flask import Flask, request, jsonify

from transformers import AutoModelForSequenceClassification, AutoTokenizer

def download_model():
    url = "https://drive.google.com/uc?export=download&id=1kQMzAoxTw038szO9q0hmpOW2ZFzFhcGF"
    output_path = "saved_model.zip"

    if not os.path.exists("saved_model"):
        print("Downloading model...")
        gdown.download(url, output_path, quiet=False)
        extract_model()
    else:
        print("Model already downloaded.")

def extract_model():
    print("Extracting model...")
    with zipfile.ZipFile("saved_model.zip", "r") as zip_ref:
        zip_ref.extractall("saved_model")
    print("Model extracted.")

# Download and extract model
download_model()

# Handle double folder issue (`saved_model/saved_model`)
model_path = "saved_model"
if os.path.exists(os.path.join(model_path, "saved_model")):
    model_path = os.path.join(model_path, "saved_model")

print(f"Loading model from: {model_path}")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

URGENCY_MAP = {0: "Low", 1: "Medium", 2: "High"}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    complaint_text = data.get("complaint")

    if not complaint_text:
        return jsonify({"error": "No complaint text provided"}), 400

    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    urgency = URGENCY_MAP.get(predicted_class, "Unknown")

    return jsonify({"complaint": complaint_text, "urgency": urgency})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port
    app.run(host="0.0.0.0", port=port, debug=True)
