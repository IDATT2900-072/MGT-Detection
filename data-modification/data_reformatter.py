import pandas as pd
import csv
import os
import re

from data_processing import word_length_of, completion_bar


def reformat_supervised_learning(source_csv_path, real_label, generated_label, title, target_dir_path,
                                 target_file_name):
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
    fields = ['title', 'label', 'text']
    formatted_data = []

    for i in range(len(data)):
        formatted_data.append([data[title][i], 0, data[real_label][i]])
        formatted_data.append([data[title][i], 1, data[generated_label][i]])

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


def deduplicate_csv(source_csv_path, target_file_name, target_dir_path="./"):
    data = pd.read_csv(source_csv_path, engine="python")
    deduplicated = data.drop_duplicates(subset=['title'])
    deduplicated.to_csv(target_dir_path + "/" + target_file_name + ".csv", index=False)


def recount_words_csv(column_pairs: [(str, str)], source_csv_path, target_dir_path, target_file_name, delim=','):
    data = pd.read_csv(source_csv_path, engine="python", delimiter=delim)
    fields = list(data.columns)
    new_data = []

    for i, row in data.iterrows():
        completion_bar(i, data.size, f'Recounting rows')
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


# Execution queue
# reformat_supervised_learning(source_csv_path="../../datasets/origins/GPT-wiki-intro.csv",
#                              real_label="wiki_intro",
#                              generated_label="generated_intro",
#                              title='title',
#                              target_dir_path="../../datasets/human-vs-machine",
#                              target_file_name="wiki-labeled")

#
# deduplicate_csv(source_csv_path='../../datasets/origins/research-abstracts/research_abstracts_final.csv',
#                 target_dir_path='../../datasets/origins/research-abstracts/',
#                 target_file_name='research_abstracts-deduplicated')

# clean_text_column(dirty_columns=["real_abstract", 'generated_abstract', 'title'],
#                   cleaning_func=cleanup_newlines,
#                   source_csv_path="../../datasets/origins/research-abstracts/research_abstracts-uniform.csv",
#                   target_dir_path="../../datasets/origins/research-abstracts",
#                   target_file_name="research_abstracts_cleaned")

# recount_words_csv(column_pairs=[('real_abstract', 'real_word_count'), ('generated_abstract', 'generated_word_count')],
#                   source_csv_path='../../datasets/origins/research-abstracts/research_abstracts_reviewed.csv',
#                   target_dir_path="../../datasets/origins/research-abstracts",
#                   target_file_name="research_abstracts_recounted-test",
#                   delim=';')

reformat_supervised_learning(source_csv_path="../../datasets/origins/research-abstracts/research_abstracts-final.csv",
                             real_label="real_abstract",
                             generated_label="generated_abstract",
                             title='title',
                             target_dir_path="../../datasets/human-vs-machine",
                             target_file_name="research-abstracts-labeled")
