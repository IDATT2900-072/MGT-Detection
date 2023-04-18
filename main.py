from models.fine_tuner import FineTuner
import datasets
import wandb

ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")

model_name = "roberta-base-openai-detector"

wandb.init(project="IDATT2900-072",
           config = {
            'base-model': model_name,
            'dataset': ds['train'].config_name,
           })

print("="*150)
print("Initiating trainer...")
tuner = FineTuner(model_name, ds, num_epochs=1, logging_steps=20)
print("="*150)
print("Starting training...")
tuner.train()
