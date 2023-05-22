from models.fine_tuner import FineTuner
import datasets

ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")

print("Initiating...")
model_name = "andreas122001/roberta-academic-detector"
tuner = FineTuner(model_name, ds, max_tokenized_length=512, do_wandb_logging=False)

print("Testing...")
results = tuner.test(ds)
print(results)


