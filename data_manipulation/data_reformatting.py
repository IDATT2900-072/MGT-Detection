import pandas as pd
import csv
import os
import re

from data_processing import word_length_of, completion_bar


def reformat_supervised_learning(source_csv_path, title, real_label, real_word_count, generated_label, generated_word_count, target_dir_path,
                                 target_file_name):
    """
    Reformats an origin dataset into a two-column dataset consisting of a text and its corresponding binary class
    label, omitting redundant features of the origin dataset.

    Parameters
    ----------
    source_csv_path : str
        Path to origin dataset in csv format.
    title : str
        title of data point
    real_label : str
        Column name of the real entries in origin dataset.
    real_word_count : str
        Column name of the real texts' word_count
    generated_label : str
        Column name of the generated entries in origin dataset.
    generated_word_count : str
        Column name of the generated texts' word_count
    target_dir_path : str
        Path to the target directory for the reformatted dataset.
    target_file_name : str
        Name of the reformatted dataset.
    """
    data = pd.read_csv(source_csv_path, engine="python")
    fields = ['title', 'label', 'text', 'word_count']
    formatted_data = []

    for i in range(len(data)):
        formatted_data.append([data[title][i], 0, data[real_label][i], data[real_word_count][i]])
        formatted_data.append([data[title][i], 1, data[generated_label][i], data[generated_word_count][i]])

    os.makedirs(target_dir_path, exist_ok=True)
    with open(target_dir_path + "/" + target_file_name + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(formatted_data)

    print(f"Reformatting complete. Number of entries in reformatted dataset: {len(formatted_data)}")


def clean_text_column(dirty_columns, cleaning_func, source_csv_path, target_dir_path, target_file_name):
    """
    Replaces all text-entries in a csv-column with the using the passed cleaning function.

    Parameters
    ----------
    dirty_columns : list[str]
        A list of strings, containing the columns be cleaned.
    cleaning_func : function
        A cleaning function which takes in a single string and returns the cleaned string.
    source_csv_path : str
        Path to origin dataset in csv format.
    target_dir_path : str
        Path to the target directory for the reformatted dataset.
    target_file_name : str
        Name of the reformatted dataset.
    """
    data = pd.read_csv(source_csv_path, engine="python")

    for dirty_column in dirty_columns:
        data[dirty_column] = data[dirty_column].apply(cleaning_func)

    os.makedirs(target_dir_path, exist_ok=True)
    data.to_csv(target_dir_path + "/" + target_file_name + ".csv", encoding='utf-8', index=False)

    print(f"Column(s) cleaned.")


def deduplicate_csv(source_csv_path, unique_key, target_file_name, target_dir_path="./", write_csv=True):
    data = pd.read_csv(source_csv_path, engine="python")

    num_dupes = data[unique_key].size - len(data[unique_key].unique())
    print(f'Found {num_dupes} duplicates.')

    if write_csv:
        path = target_dir_path + "/" + target_file_name + ".csv"
        deduplicated = data.drop_duplicates(subset=[unique_key])
        deduplicated.to_csv(path, index=False)
        print(f'Deduplicated csv at {path}')


def recount_words_csv(column_pairs: [(str, str)], source_csv_path, target_dir_path, target_file_name, delim=','):
    data = pd.read_csv(source_csv_path, engine="python", delimiter=delim)
    fields = list(data.columns)
    new_data = []

    for i, row in data.iterrows():
        completion_bar(i, data[fields[0]].size, f'Recounting rows')
        new_data.append([row[field] for field in fields])

        if i == 0:
            continue

        for (text_column, count_column) in column_pairs:
            text_index, count_index = 0, 0

            for j, field in enumerate(fields):
                if field == count_column:
                    count_index = j
                elif field == text_column:
                    text_index = j

            new_data[i][count_index] = word_length_of(row[text_index])

    os.makedirs(target_dir_path, exist_ok=True)
    with open(target_dir_path + "/" + target_file_name + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(new_data)

    print(f"\nColumn(s) recounted.")