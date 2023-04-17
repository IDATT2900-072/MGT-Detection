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
    metric4 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels)["precision"]
    recall = metric3.compute(predictions=predictions, references=labels)["recall"]
    f1 = metric4.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Fine-tunes a pre-trained model on a specific dataset
class FineTuner:
    def __init__(self, model_name, dataset, num_train_samples=None, num_eval_samples=None, max_tokenized_length=None):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.max_tokenized_length = max_tokenized_length

        # Initialize trainer
        self.trainer = self.init_trainer()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding='max_length', truncation=True, max_length=self.max_tokenized_length)

    def init_trainer(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)

        # Training and evaluation datasets
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

        # Select subsets for faster training
        if (self.num_train_samples):
            train_dataset = train_dataset.select(range(self.num_train_samples))
        if (self.num_eval_samples):
            eval_dataset = eval_dataset.select(range(self.num_eval_samples))

        training_args = TrainingArguments(output_dir="./outputs/"+self.model.config._name_or_path + "-tuned",
                                          logging_dir="./logs",
                                          logging_steps=100,
                                          logging_first_step=True,
                                          evaluation_strategy="steps", 
                                          save_strategy='epoch',
                                          optim='adamw_torch', 
                                          num_train_epochs=5,
                                          auto_find_batch_size=True,
        )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
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
