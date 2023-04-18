from models.fine_tuner import FineTuner
import datasets
import wandb

ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")
ds_mlt = 0.1
ds['train'] = ds['train'].select(range(int(len(ds['train'])*ds_mlt)))
ds['test'] = ds['test'].select(range(int(len(ds['train'])*ds_mlt)))
ds['validation'] = ds['validation'].select(range(int(len(ds['train'])*ds_mlt)))

model_name = "bigscience/bloomz-560m"

wandb.init(project="IDATT2900-072",
           config = {
            'base-model': model_name,
            'dataset': ds['train'].config_name,
           })

print("="*150)
print("Initiating trainer...")
tuner = FineTuner(model_name, ds, num_epochs=0.1, logging_steps=200, max_tokenized_length=512)
print("="*150)
print("Starting training...")
tuner.train()
