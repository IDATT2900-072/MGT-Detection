from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
model_name = "andreas122001/bloomz-560m-wiki-detector"
dataset_name = "wiki_labeled"

# Load dataset
dataset = datasets.load_dataset("NicolaiSivesind/human-vs-machine", dataset_name, split="test")

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
trues = [0, 0] # negative, positive
falses = [0, 0] # negative, positive

dataloader = DataLoader(dataset, batch_size=8)
for i, batch in enumerate(tqdm(dataloader)):

    predicted_labels = predict(batch['text'])
    real_labels = batch['label']

    # Assert reals and falses for positive and negative results
    for real, pred in zip(real_labels, predicted_labels):
        if real == pred:
            trues[pred] += 1
        else:
            falses[pred] += 1

    if i%100 == 0:
        print("\n" +str(trues),str(falses))
        
# For debugging
print("trues (n/p):" + str(trues))
print("falses (n/p):" + str(falses))

# Gather results in array and normalize
results = np.array([
    [trues[0], falses[1]],
    [falses[0], trues[1]]
    ])
results = np.round((results / np.sum(results))*100, 2)

# Plot as heatmap
labels_x = ["0", "1"]
labels_y = ["0", "1"]

fig, ax = plt.subplots()
im = ax.imshow(results, cmap="Blues")

ax.set_title("Bloomz-560m on wiki_labeled")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(np.arange(len(labels_y)), labels=labels_y)
ax.set_yticks(np.arange(len(labels_x)), labels=labels_x)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")

for i in range(len(labels_x)):
    for j in range(len(labels_y)):
        if results[i, j] < 20:
            color = "black"
        else:
            color = "w"
        text = ax.text(j, i, (str(results[i, j]) + "%"),
                       ha="center", va="center", color=color)
fig.tight_layout()
plt.savefig(model_name + "_result.png", bbox_inches='tight')