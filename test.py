from model.fine_tuner import FineTuner
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

ds = datasets.load_dataset("dataset", "wiki_labeled")
ds = ds.rename_column("class label", "labels")

# print(ds['test'][0])

# device = "cuda" if torch.cuda.is_available else "cpu"

# tokenizer = AutoTokenizer.from_pretrained("roberta-large-openai-detector", max_length=512)
# model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector").to(device)
# model.eval()

# pipe = pipeline("text-classification", device=0, model=model, tokenizer=tokenizer)

# print(pipe(ds['test'].select(range(10))['text']))

print("="*150)
tuner = FineTuner("roberta-base-openai-detector", ds)
print("="*150)
print("Starting training...")
tuner.train()
