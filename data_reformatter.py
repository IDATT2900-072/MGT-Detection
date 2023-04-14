import pandas as pd
import csv
import os


def reformat(source_csv_path, real_label, generated_label, target_dir_path, target_file_name):
    """
    Reformat an origin dataset into a two-column dataset consisting of a text and its corresponding binary class
    label, omitting redundant features of the origin dataset.

        :param str source_csv_path: Path to origin dataset in csv format.
        :param str real_label: Column name of the real entries in origin dataset.
        :param str generated_label: Column name of the generated entries in origin dataset.
        :param str target_dir_path: Path to the target directory for the reformatted dataset.
        :param str target_file_name: Name of the reformatted dataset.
    """
    data = pd.read_csv(source_csv_path, engine="python")
    fields = ['class label', 'text']
    formatted_data = []

    for i in range(len(data)):
        formatted_data.append([0, data[real_label][i]])
        formatted_data.append([1, data[generated_label][i]])

    os.makedirs(target_dir_path, exist_ok=True)
    with open(target_dir_path + "/" + target_file_name + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(formatted_data)

    print(f"Reformatting complete. Number of entries in reformatted dataset: {len(formatted_data)}")

# Execution queue
reformat(source_csv_path="../../Datasets/GPT-wiki-intro.csv",
        real_label="wiki_intro",
        generated_label="generated_intro",
        target_dir_path="./dataset",
        target_file_name="wiki-labeled")
