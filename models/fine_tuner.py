import evaluate
import numpy as np
from datasets import Dataset
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from typing import Dict, List, Union
import os
import re

class FineTuner:
    """Fine-tunes a pre-trained model on a specific dataset"""

    def __init__(self, model_name, dataset, 
                 num_epochs=5, 
                 max_tokenized_length=None, 
                 logging_steps=500, 
                 do_wandb_logging=True,
                 remove_white_spaces=False,
                 ):
        self.test_dataset = None
        self.dataset = dataset
        self.labels = dataset['train'].features['label'].names
        self.id2label = {0: self.labels[0], 1: self.labels[1]}
        self.label2id = {self.labels[0]: 0, self.labels[1]: 1}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        num_labels=2,
									                                    low_cpu_mem_usage=True,
                                                                        label2id=self.label2id,
                                                                        id2label=self.id2label)
        self.max_tokenized_length = max_tokenized_length
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.do_wandb_logging = do_wandb_logging
        self.seed = 42

        # Initialize trainer
        self.trainer = self.init_trainer()

        # Pre-process dataset
        if remove_white_spaces:
            print("Removing white-spaces")
            def remove_newline(dataset):
                dataset['text'] = re.sub(r'\s+', ' ', dataset['text'])        
                return dataset

            for split_name, split in zip(self.dataset.keys(), self.dataset.values()):
                self.dataset[split_name] = split.map(remove_newline)

        # Initialize Weights and Biases
        if self.do_wandb_logging:
            self.wandb = wandb.init(project="IDATT2900-072",
                    config={
                        'base_model': model_name,
                        'dataset': dataset['train'].config_name,
                        'train_dataset_size': len(dataset['train']),
                        'eval_dataset_size': len(dataset['validation']),
                        'max_tokenized_length': self.max_tokenized_length,
                    },
                    tags=[("no-white-space" if remove_white_spaces else "white-space")]
                )
        else:
            os.environ["WANDB_DISABLED"] = "true"
                       
    def tokenize_function(self, examples):
        return self.tokenizer(text_target=examples["text"],
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_tokenized_length)

    def init_trainer(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)

        # Training, validation and test datasets (tokenized and shuffled)
        train_dataset = tokenized_datasets["train"].shuffle(seed=self.seed)
        validation_dataset = tokenized_datasets["validation"].shuffle(seed=self.seed)
        self.test_dataset = tokenized_datasets["test"].shuffle(seed=self.seed)

        # E.g "bloomz-560m-wiki_labeled-detector"
        save_name = self.model.config._name_or_path.split("/")[-1] + "-" + self.dataset['train'].config_name + "-detector"

        training_args = TrainingArguments(output_dir="./outputs/" + save_name,
                                          logging_dir="./logs",
                                          logging_steps=self.logging_steps,
                                          logging_first_step=True,
                                          evaluation_strategy="steps",
                                          save_strategy='epoch',
                                          optim='adamw_torch',
                                          num_train_epochs=self.num_epochs,
                                          auto_find_batch_size=True,
                                          )
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=self.__compute_metrics,
        )

        return trainer

    def train(self):
        self.trainer.train()

    def test(self, dataset: Dataset = None) -> Dict[str, float]:
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
            test_dataset = dataset.map(self.tokenize_function, batched=True)["test"].shuffle(self.seed)

        # Prefix for WandB - to destinguish between the tests if running multiple tests
        prefix = "test" if not dataset else "test_" + test_dataset.config_name.replace("_", "-")
        metrics = self.trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix=prefix)

        # If using default, we still want to log as test/"dataset"_"metric"
        if not dataset and self.do_wandb_logging:
            prefix = "test/" + test_dataset.config_name.replace("_", "-")
            res = {prefix + str(key).replace("test", ""): val for key, val in metrics.items()}
            wandb.log(res)
        return metrics
    
    def classify(self, text):
        # Initialize pipeline
        classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0)
        return classifier(text)

    def predict(self, data: Union[str, List[str]]) -> torch.Tensor:
        """
        Generates a prediction for the data and returns probabilities as a tensor.
        """
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Encode input text and labels
        encoding = self.tokenizer(data, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

        # Execution
        with torch.no_grad():
            outputs = self.model(encoding['input_ids'])
            logits = outputs.logits.squeeze()

        # Calculate probabilities
        probabilities = torch.softmax(logits.cpu(), dim=-1).detach().numpy()
        return probabilities.tolist()
    
    def get_labels(self):
        return self.labels

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
