from models.fine_tuner import FineTuner
import datasets
import sys
from data_manipulation.data_processing import remove_white_spaces

dataset_name = "wiki_labeled"
ds_mlt = 0.1 # how much of the dataset to use - default 10%
model_name = "roberta-base"
use_cross = False

def smallify(dataset, relative_size):
    for split_name, split in zip(dataset.keys(), dataset.values()):
        dataset[split_name] = split.select(range(int(len(split)*relative_size)))

def concatinate_ds(ds1, ds2):
    for split_name in ds1.keys():
        ds1[split_name] = datasets.concatenate_datasets([ds1[split_name], ds2[split_name]])

# Parses the job-name from the sbatch script for running bloomz-tuning
if len(sys.argv) == 2:
    print("Using arguments from sbatch")
    args = sys.argv[1].split("-")
    if len(args) >= 3:
        if args[0] == "rob":
            model_name = "roberta-base"
        else:
            model_name = "bigscience/bloomz-" + args[0]
        
        use_cross = False
        if args[1] == "wiki":
            dataset_name = "wiki_labeled"
        elif args[1] == "abs":
            dataset_name = "research_abstracts_labeled"
        elif args[1] == "cross":
            dataset_name = "cross"
            use_cross = True
        ds_mlt = 1/float(args[2])
        print("Using arguments: ", model_name, dataset_name, ds_mlt)
    else:
        raise Exception("Expected following format for input argument: 'model-dataset-datasetSize-xxx")

if use_cross:
    print("Using mixed dataset")
    ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")
    ds2 = datasets.load_dataset("NicolaiSivesind/human-vs-machine", "research_abstracts_labeled")

    smallify(ds, 0.05)
    smallify(ds2, 0.5)
    remove_white_spaces(ds2)

    concatinate_ds(ds, ds2)
else:
    ds = datasets.load_dataset("NicolaiSivesind/human-vs-machine", dataset_name)

smallify(ds, ds_mlt)

print(ds)

# This dataset must be pre-processed - removing newlines and white-spaces
if dataset_name == "research_abstracts_labeled":
    do_remove_white_spaces = True
else: 
    do_remove_white_spaces = False

print("="*150)
print("Initiating trainer...")
logging_steps = int(max(1, len(ds['train'])/(50*8))) # assuming a batch_size of 8, will make sure every epoch has 50 evaluations
tuner = FineTuner(model_name, ds, 
                  num_epochs=1, 
                  logging_steps=logging_steps, 
                  max_tokenized_length=512, 
                  do_wandb_logging=True,
                  remove_white_spaces=do_remove_white_spaces
                  )
print("="*150)
print("Starting training...")
tuner.train()
print("Starting tests...")
tuner.test()
