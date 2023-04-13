import evaluate
import numpy as np
import datasets as ds
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = ds.load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)


class FineTuner:
    def __init__(self, model_name, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = self.init_trainer()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def init_trainer(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        return trainer

    def train(self):
        self.trainer.train()

    def get_prediction(self, text):
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.to(self.trainer.model.device) for k, v in encoding.items()}

        outputs = self.model(**encoding)

        logits = outputs.logits

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)

        if label == 1:
            return {
                'sentiment': 'Generated',
                'probability': probs[1]
            }
        else:
            return {
                'sentiment': 'Real',
                'probability': probs[0]
            }
