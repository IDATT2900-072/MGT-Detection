import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Load the dataset and split into train and test sets
dataset = load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")
train_dataset = dataset["train"]
validate_dataset = dataset["validate"]
test_dataset = dataset["test"]

dataset = load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled", use_auth_token="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")


# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="epoch",
)


# Compute evaluation metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    return {"accuracy": accuracy}


# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# Add this function to your existing script
def inference(text):
    # Preprocess input text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate probabilities and find the most likely label
    probabilities = torch.softmax(logits, dim=-1)
    most_likely_label_index = torch.argmax(probabilities, dim=-1).item()
    labels = ["Human produced text", "Machine generated text"]
    most_likely_label = labels[most_likely_label_index]
    score = probabilities[0, most_likely_label_index].item()

    return most_likely_label, score


trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Train and evaluate the model as before, and then use the inference function:
text = "This is an example text for classification."
label, score = inference(text)
print(f"Predicted label: {label}, Classification score: {score}")
