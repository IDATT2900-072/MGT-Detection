from models.fine_tuner import FineTuner
import datasets

ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")
ds_mlt = 0.1
ds['train'] = ds['train'].select(range(int(len(ds['train'])*ds_mlt)))
ds['test'] = ds['test'].select(range(int(len(ds['test'])*ds_mlt)))
ds['validation'] = ds['validation'].select(range(int(len(ds['validation'])*ds_mlt)))

model_name = "bigscience/bloomz-560m"

print("="*150)
print("Initiating trainer...")
tuner = FineTuner(model_name, ds, num_epochs=1, logging_steps=int(1000*ds_mlt), max_tokenized_length=512)
print("="*150)
print("Starting training...")
tuner.train()
print("Starting tests...")
tuner.test()