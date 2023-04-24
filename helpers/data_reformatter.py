import pandas as pd
import csv
import os
import re


def reformat_for_classification(source_csv_path, real_label, generated_label, target_dir_path, target_file_name):
    """
    Reformats an origin dataset into a two-column dataset consisting of a text and its corresponding binary class
    label, omitting redundant features of the origin dataset.

    Parameters
    ----------
    source_csv_path : str
        Path to origin dataset in csv format.
    real_label : str
        Column name of the real entries in origin dataset.
    generated_label : str
        Column name of the generated entries in origin dataset.
    target_dir_path : str
        Path to the target directory for the reformatted dataset.
    target_file_name : str
        Name of the reformatted dataset.
    """
    data = pd.read_csv(source_csv_path, engine="python")
    fields = ['label', 'text']
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


def cleanup_newlines(text):
    """
    Cleans a text, replacing all newlines with double newline if preceding character is '.', '!', or "?", else replaces
    with a single space character. All sequences of whitespaces are replaced with a space character unless it's a
    newline.

    Parameters
    ----------
    text : str
        String of text to be cleaned.

    Returns
    -------
    str
        The cleaned version of the text.

    """

    # Replace all newlines which does not succeed '\n', '.', '!' or '?', and does not precede another newline, with a
    # space character.
    clean = re.sub(r'(?<![.!?\n])\n(?!\n)', ' ', text)

    # Replace all newlines which succeeds a '.', '!' or '?' and does not precede another newline, with a double newline.
    clean = re.sub(r'(?<=[.!?])\n(?!\n)', '\n\n', clean)

    # Replace all non-newline sequences of whitespace with a single space character.
    clean = re.sub(r'[^\S\n]+', ' ', clean)

    # Remove all non-newline characters succeeding a newline character.
    clean = re.sub(r'(\n[^\S\n]+)', '\n', clean)

    # Remove any whitespace preceding first non-whitespace character of the text.
    clean = re.sub(r'^\s+', '', clean)

    # Remove everything after last '.', '!', or '?'.
    clean = re.sub(r'([.!?])(?:(?!\1).)*$', r'\1', clean)

    return clean


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


# Execution queue
# reformat_for_classification(source_csv_path="../../datasets/origins/GPT-wiki-intro.csv",
#                             real_label="wiki_intro",
#                             generated_label="generated_intro",
#                             target_dir_path="../../datasets/human-vs-machine",
#                             target_file_name="wiki-labeled")


# clean_text_column(dirty_columns=["real_abstract", 'generated_abstract'],
#                   cleaning_func=cleanup_newlines,
#                   source_csv_path="../../datasets/origins/research-abstracts/research_abstracts.csv",
#                   target_dir_path="../../datasets/origins/research-abstracts",
#                   target_file_name="research_abstracts_cleaned")

reformat_for_classification(source_csv_path="../../datasets/origins/research-abstracts/research_abstracts_cleaned.csv",
                            real_label="real_abstract",
                            generated_label="generated_abstract",
                            target_dir_path="../../datasets/human-vs-machine",
                            target_file_name="research-abstracts")

