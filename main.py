from models.fine_tuner import FineTuner
import datasets
import sys

dataset_name = "wiki_labeled"
ds_mlt = 0.1 # how much if the dataset to use - default 10%
model_name = "bigscience/bloomz-560m"

# Parses the job-name from the sbatch script for running bloomz-tuning
if len(sys.argv) == 2:
    args = sys.argv[1].split("-")
    if len(args) >= 3:
        model_name = "bigscience/bloomz-" + args[0]
        dataset_name = "research_abstracts_labeled" if args[1] == "abs" else "wiki_labeled"
        ds_mlt = 1/float(args[2])
        print("Using arguments: ", model_name, dataset_name, ds_mlt)
    else:
        raise Exception("Expected following format for input argument: 'model-dataset-datasetSize-xxx")

ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", dataset_name)
ds['train'] = ds['train'].select(range(int(len(ds['train'])*ds_mlt)))
ds['test'] = ds['test'].select(range(int(len(ds['test'])*ds_mlt)))
ds['validation'] = ds['validation'].select(range(int(len(ds['validation'])*ds_mlt)))

print("="*150)
print("Initiating trainer...")
tuner = FineTuner(model_name, ds, num_epochs=1, logging_steps=int(500*ds_mlt), max_tokenized_length=512, do_wandb_logging=False)
print("="*150)
print("Starting training...")
tuner.train()
print("Starting tests...")
tuner.test()