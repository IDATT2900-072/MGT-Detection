from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from time import sleep
import sys

model_name = "andreas122001/bloomz-560m-wiki-detector"
dataset_name = "wiki_labeled"
num_data = None

# Parses the job-name from the sbatch script for running bloomz-tuning
# eval-rob-wiki-wiki
if len(sys.argv) == 2:
    print("Using arguments from sbatch")
    args = sys.argv[1].split("-")
    if len(args) >= 3:
        model_name = "andreas122001/"
        if args[1] == "rob":
            model_name += "roberta"
        else:
            model_name += "bloomz-" + args[1]
        if args[2] == "wiki":
            model_name += "-wiki-detector"
        elif args[2] == "abs":
            model_name += "-academic-detector"
    
        dataset_name = "research_abstracts_labeled" if args[3] == "abs" else "wiki_labeled"
        print("Using arguments: ", model_name, dataset_name)
    else:
        raise Exception("Expected following format for input argument: 'eval-model-trainDataset-testDataset-xxx")

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
print()

model_name = model_name.split("/")[-1]
print(f"{model_name} on {dataset_name}")
for i in trues:
    print(i)
for i in falses:
    print(i)
