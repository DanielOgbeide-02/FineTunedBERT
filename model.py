from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

# Load dataset
dataset = load_dataset("csv", data_files="current_dataset.csv")  # Change to your actual file

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize text
def tokenize_function(example):
    return tokenizer(example["Complaints"], truncation=True, padding="max_length")

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define mapping for urgency levels
URGENCY_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Function to convert urgency labels
def convert_labels(example):
    urgency_str = example["Urgency"].strip()  # Remove extra spaces
    if urgency_str not in URGENCY_MAP:
        raise ValueError(f"Unexpected urgency value: '{urgency_str}'")
    return {"labels": URGENCY_MAP[urgency_str]}

# Apply label conversion
tokenized_datasets = tokenized_datasets.map(convert_labels)

# Split dataset into train and test (80% train, 20% test)
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Load pre-trained model for text classification (BERT)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Function to test on new complaints
def predict_urgency(complaint_text):
    inputs = tokenizer(complaint_text, return_tensors="pt", truncation=True, padding="max_length")
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    urgency_labels = {0: "Low", 1: "Medium", 2: "High"}
    return urgency_labels[prediction]

# Example test
test_complaint = "My internet has been down for hours and I need it for work!"
predicted_urgency = predict_urgency(test_complaint)
print(f"Predicted urgency: {predicted_urgency}")
