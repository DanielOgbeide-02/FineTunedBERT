import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your trained model and tokenizer
model_path = r"C:\Users\danie\PycharmProjects\FineTuneBert\saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Mapping numbers to urgency labels
URGENCY_MAP = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# Function to predict complaint category
def predict_complaint(complaint_text):
    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class number back to its urgency label
    return URGENCY_MAP.get(predicted_class, "Unknown")  # Default to "Unknown" if not found

# Test cases
test_complaints = [
    "The wifi in my wing is not working.",
    "The socket in my room just exploded.",
    "The taps in the bathroom gushes out much.",
]

for complaint in test_complaints:
    prediction = predict_complaint(complaint)
    print(f"Complaint: {complaint}\nPredicted Urgency: {prediction}\n")
