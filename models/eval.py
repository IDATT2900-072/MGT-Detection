from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from time import sleep

model_name = "andreas122001/bloomz-560m-wiki-detector"
dataset_name = "wiki_labeled"
num_data = None

print(f"\nModel: {model_name}\nDataset: {dataset_name}\n")

# Load dataset
dataset = datasets.load_dataset("NicolaiSivesind/human-vs-machine", dataset_name, split="test")
if num_data:
    dataset = dataset.select(range(num_data))

# Load model and tokenizer into pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Define predict function
def predict(batch):
    encoding = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    outputs = model(**encoding)
    logits = outputs.logits.squeeze()
    pred = torch.softmax(logits.cpu(), dim=-1).detach().numpy()

    return np.argmax(pred, -1)

# Perform tests
trues = [0, 0] # [negatives, positives] - all true negative and positive predicions
falses = [0, 0] # [negatives, positives] - all false negative and positive predicions

num_prints = 50 # how many times to print intermediate results
dataloader = DataLoader(dataset, batch_size=3)
for i, batch in enumerate(tqdm(dataloader)):

    predicted_labels = np.array([predict(batch['text'])]).reshape(-1)
    real_labels = batch['label'].detach().numpy()

    # Assert reals and falses for positive and negative results
    for real, pred in zip(real_labels, predicted_labels):
        if real == pred:
            trues[pred] += 1
        else:
            falses[pred] += 1

    if i%max(1, len(dataloader) // num_prints) == 0:
        print("\n" +str(trues),str(falses))

# Print result - sleep so tqdm doesn't overwrite results
sleep(1)
print("\nResults: (t/f)")
print("trues = " + str(trues))
print("falses = " + str(falses))