import evaluate
import numpy as np
from datasets import Dataset
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict


class FineTuner:
    """Fine-tunes a pre-trained model on a specific dataset"""
    def __init__(self, model_name, dataset, 
                 num_epochs=5, 
                 max_tokenized_length=None, 
                 logging_steps=500,
                 ):
        self.dataset = dataset
        self.labels = dataset['train'].features['label'].names
        self.id2label = {0: self.labels[0], 1: self.labels[1]}
        self.label2id = {self.labels[0]: 0, self.labels[1]: 1}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                        num_labels=2, 
                                                                        low_cpu_mem_usage=True,
                                                                        label2id=self.label2id,
                                                                        id2label=self.id2label,
                                                                        )
        self.max_tokenized_length = max_tokenized_length
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps

        # Initialize trainer
        self.trainer = self.init_trainer()

        # Initialize Weights and Biases
        wandb.init(project="IDATT2900-072",
           config = {
            'base_model': model_name,
            'dataset': dataset['train'].config_name,
            'train_dataset_size': len(dataset['train']),
            'eval_dataset_size': len(dataset['validation']),
        })

    def tokenize_function(self, examples):
        return self.tokenizer(text_target=examples["text"],
                              padding='max_length', 
                              truncation=True, 
                              max_length=self.max_tokenized_length,
                              )

    def init_trainer(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)

        # Training, validation and test datasets (tokenized and shuffled) 
        self.train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        self.validation_dataset = tokenized_datasets["validation"].shuffle(seed=42)
        self.test_dataset = tokenized_datasets["test"].shuffle(seed=42)

        # E.g "bloomz-560m-wiki_labeled-27000"
        save_name = self.model.config._name_or_path + "-" + self.dataset['train'].config_name + "-" + str(len(self.dataset['train']))

        training_args = TrainingArguments(output_dir="./outputs/"+save_name,
                                          logging_dir="./logs",
                                          logging_steps=self.logging_steps,
                                          logging_first_step=True,
                                          evaluation_strategy="steps",
                                          save_strategy='steps',
                                          save_steps=int(2*self.logging_steps),
                                          optim='adamw_torch', 
                                          num_train_epochs=self.num_epochs,
                                          auto_find_batch_size=True,
        )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            compute_metrics=self.__compute_metrics,
        )

        return trainer

    def train(self):
        self.trainer.train()

    def test(self, dataset: Dataset=None) -> Dict[str, float]:
        """
        Tests the model on a dataset and logs metrics to WandB.

            params:
                dataset (`Dataset`, *optional*):
                    The dataset to test. If none is provided, use the dataset of the test-dataset of the tuner.
            returns:
                dict[str, float]:
                    A dictionary containing the metrics
        """

        # If none provided, use default test-dataset
        if not dataset:
            test_dataset = self.test_dataset
        # If using other dataset, we have to tokenize it
        else: 
            # TODO: Seems this is taking waaay too much memory. 
            # Might need to clear default dataset from memory first somehow. 
            # More testing is needed though - but testing memory on a cluster is not so much fun.
            test_dataset = dataset.map(self.tokenize_function, batched=True)["test"]

        # Prefix for WandB - to destinguish between the tests if running multiple tests
        prefix = "test" if not dataset else "test_" + test_dataset.config_name.replace("_","-")
        metrics = self.trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix=prefix)

        # If using default, we still want to log as test/"dataset"_"metric"
        if not dataset:
            prefix = "test/" + test_dataset.config_name.replace("_","-")
            res = {prefix + str(key).replace("test",""): val for key, val in metrics.items()}
            wandb.log(res)
        return metrics

    def classify(self, text):
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

    def __compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
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