import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets as ds


class ZeroShooter:

    def __init__(self, model_name, dataset):
        # Load dataset and initialize label domain
        self.dataset = dataset
        self.labels = ["Generated", "Real"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def classify(self):
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        subset = self.dataset["test"].select(range(10))

        # Zero-shot classification on the selected dataset
        for data_point in subset:
            text = data_point["text"]

            # Encode input text and labels
            inputs = self.tokenizer.encode_plus(text, return_tensors="pt", truncation=True,
                                                padding="max_length")
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(self.labels, return_tensors="pt", padding="max_length",
                                         truncation=True)

            # Move inputs and targets to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            # Execution
            with torch.no_grad():
                outputs = self.model(**inputs, labels=targets["input_ids"])
                logits = outputs.logits

            # Calculate probabilities and find the most likely label
            probabilities = torch.softmax(logits, dim=-1)
            most_likely_label_index = torch.argmax(probabilities, dim=-1).item()
            most_likely_label = self.labels[most_likely_label_index]
            score = probabilities[0, most_likely_label_index].item()

            print(f"Text: {text[:100]}\nPredicted label: {most_likely_label}, Classification score: {score}\n")
