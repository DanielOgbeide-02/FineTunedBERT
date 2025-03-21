import os
import gdown
import zipfile
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def download_model():
    url = "https://drive.google.com/uc?export=download&id=1kQMzAoxTw038szO9q0hmpOW2ZFzFhcGF"
    output_path = "saved_model.zip"

    if not os.path.exists("saved_model"):  # Avoid re-downloading
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

# Download and extract the model before loading it
download_model()

# Ensure model is loaded from the correct path
model_path = "saved_model"

# Check if the model path contains a subfolder after extraction
possible_subfolders = [f.path for f in os.scandir(model_path) if f.is_dir()]
if possible_subfolders:
    model_path = possible_subfolders[0]  # Use the first subfolder

print(f"Loading model from: {model_path}")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()  # Set model to evaluation mode

# Urgency Mapping
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
