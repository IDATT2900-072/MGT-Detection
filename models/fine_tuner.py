import evaluate
import numpy as np
import datasets as ds
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# Function for computing evaluation metrics
def compute_metrics(eval_pred):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("precision")
    metric3 = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels)["precision"]
    recall = metric3.compute(predictions=predictions, references=labels)["recall"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


# Fine-tunes a pre-trained model on a specific dataset
class FineTuner:
    def __init__(self, model_name, dataset, num_train_samples=None, num_eval_samples=None):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = self.init_trainer()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def init_trainer(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)

        # Training and evaluation datasets
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

        training_args = TrainingArguments(output_dir="test_trainer", 
                                          evaluation_strategy="epoch", 
                                          optim='adamw_torch', 
                                          auto_find_batch_size=True,
                                          num_train_epochs=5)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        return trainer

    def train(self):
        self.trainer.train()

    def classify(self):
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        subset = self.dataset["test"].select(range(10))

        # Fine-tuned classification on the selected dataset
        for data_point in subset:
            text = data_point["text"]

            # Encode input text and labels
            encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
            encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

            # Execution
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits.squeeze()

            # Calculate probabilities and find the most likely label
            probabilities = torch.softmax(logits.cpu(), dim=-1)
            most_likely_label_index = torch.argmax(probabilities, dim=-1).item()
            most_likely_label = self.trainer.label_names[most_likely_label_index]
            score = probabilities[most_likely_label_index].item()

            print(f"Text: {text[:100]}\nPredicted label: {most_likely_label}, Classification Score: {score}\n")
