import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your trained model and tokenizer
model_path = r"C:\Users\danie\PycharmProjects\FineTuneBert\saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set model to evaluation model
model.eval()


# Function to predict complaint category
def predict_complaint(complaint_text):
    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class  # Adjust based on your label mapping


# Test cases
test_complaints = [
    'The main water pipe has burst, flooding my kitchen.',
    'My toilet is overflowing and won?t stop.',
    'There is no running water in the house.',
    'The sink drain is completely blocked, and water is not passing through.',
    'The shower is leaking heavily, and the bathroom is flooded.'

]

for complaint in test_complaints:
    prediction = predict_complaint(complaint)
    print(f"Complaint: {complaint}\nPredicted Class: {prediction}\n")
