import re
import random
import numpy as np
from datasets import Dataset, concatenate_datasets


def completion_bar(counter, total, text='Progress'):
    if counter % max(round(total / 100), 1) == 0:
        print('\r', f'{text}: {round(counter / total * 100)}%', end="")


def word_length_of(string):
    return len(re.findall(r'\w+', string))


def filter_and_count(dataset, word_count_column, min_word_count, max_word_count):
    if min_word_count > max_word_count:
        raise Exception(" min_word_count cannot be larger than max_word_count.")
        # Filter and select
    filtered_dataset = [match for match in dataset if min_word_count <= match[word_count_column] <= max_word_count]
    filter_domain_size = len(filtered_dataset)

    return filtered_dataset, filter_domain_size


def filter_list(data, word_count_min, word_count_max, quantity):
    """
    Filters the list, removing entries with a word_count outside the specified range, and randomly selects the desired
    quantity.

    Parameters
    ----------
    data : list[dict]
        A list of dictionaries holding at minimum a 'word_count' field.
    word_count_min : int
        Minimum accepted word count, inclusive
    word_count_max : int
        Maximum accepted word count, inclusive
    quantity : int
        The amount of selected elements the filtered list should contain.

    Returns
    -------
    list
        A list with the length of 'quantity' containing the filtered elements from the original data-modification-list. The elements
        are randomly selected from the filtered domain and a sorted in descending order based on their word_count value.
    """
    filtered_dataset, filter_domain_size = filter_and_count(data, 'word_count', word_count_min, word_count_max)
    print(f'\nFound {filter_domain_size} elements matching the filter.')

    if filter_domain_size < quantity:
        print(f"Number of filtered elements ({filter_domain_size}), is less than desired quantity ({quantity}).")
        quantity = filter_domain_size

    # Select random elements
    random.seed(42)
    filtered_dataset = random.sample(filtered_dataset, k=quantity)
    print(f'Returned list is of length {len(filtered_dataset)}.')

    # Sort in descending manner
    filtered_dataset.sort(key=lambda instance: instance['word_count'], reverse=True)
    return filtered_dataset


def count_and_reformat(dataset, count_column, retain_columns):
    """
    Counts text length in words for every data-modification point in 'column_name'-column and creates a new list with the specified
    columns to be retained, in addition to a word count column as the last column. If one of the retained columns is
    named 'word_count', it must be renamed before execution, else it will be overwritten by this functions word_count.

    Parameters
    ----------
    dataset : dict
        Dataset to extract texts from
    count_column : str
        Column name containing the texts which are to be extracted.
    retain_columns : list[str]
        A list containing the names of the columns which are to be retained in the returned list. All other columns
        will be omitted.

    Returns
    -------
    list
        A list of dictionaries containing the columns specified in the 'retain_columns'-parameter, in addition to a
        word_count-column at the end.
    """

    new_dataset = []

    # Format data_points and retrieve word_count
    total = len(dataset)
    for i, data_point in enumerate(dataset):
        if i % int(total / 100) == 0:
            print('\r', f'Counting words: {round(i / total * 100)}%', end="")
        word_count = word_length_of(data_point[count_column])
        if word_count < 50:
            continue

        new_data_point = {}
        for column in retain_columns:
            new_data_point[column] = data_point[column]
        new_data_point['word_count'] = word_count
        new_dataset.append(new_data_point)

    return new_dataset


def sample_uniform_subset(dataset, column, subset_size, start, end):
    """
    Samples a subset of selected size with a uniform distribution with respect to integer values within a set
    interval. If roof of available data points for a specific integer value is met before subset_size is reached,
    selection of data points with this integer value is skipped to ensure non-duplicate sampling, and uniform
    sampling will continue on integers with remaining (not yet selected) data points.

    Parameters
    ----------
    dataset : list[dict] | Dataset
        Dataset to be sample subset from.
    column : str
        Column name of the column with containing integer values.
    subset_size : int,
        Number of samples in returned subset.
    start : int
        Minimum integer value in returned subset - inclusive.
    end : int
        Maximum integer value in returned subset - inclusive.

    Returns
    -------
    list[dict]
        The uniformly selected subset in the format of datasets.Dataset.
    """

    dataset = list(dataset)
    subset = []

    random.seed(42)
    random.shuffle(dataset)

    # Sort data points into separate list for each integer value
    word_count_lists = {i: [] for i in range(start, end + 1)}

    for i, data_point in enumerate(dataset):
        completion_bar(i, len(dataset), text='Sorting into lists')

        word_count = data_point[column]
        if start <= word_count <= end:
            word_count_lists[word_count].append(data_point)

    print('')

    # Sample until subset size is reached or all data points have been sampled.
    while len(subset) < subset_size and len(word_count_lists) > 0:
        empty = []
        keys = list(word_count_lists.keys())

        # Randomise order for each selection round for true uniform sampling.
        random.shuffle(keys)
        for key in keys:
            completion_bar(len(subset), subset_size, text='Sampling data points')
            if len(subset) >= subset_size:
                return subset

            if len(word_count_lists[key]) == 0:
                empty.append(key)
                continue
            subset.append(word_count_lists[key].pop(0))

        # Delete integer value lists where all data points are sampled.
        for key in empty:
            del word_count_lists[key]

    return subset


def cleanup_whitespaces(text):
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


def substitute_duplicates_uniform(dataset: Dataset, source_dataset, duplicate_column :str, source_distribution_column, size_goal, start, end, seed):
    uniques = dataset.unique(duplicate_column)
    substitutes = []
    substitutions = size_goal - len(uniques)

    random.seed(seed)
    random.shuffle(source_dataset)
    source_dataset = list(source_dataset)

    # Sort data points into separate list for each integer value
    word_count_lists = {i: [] for i in range(start, end + 1)}

    for i, data_point in enumerate(source_dataset):
        completion_bar(i, len(source_dataset), text='Sorting into lists')

        word_count = data_point[source_distribution_column]
        if start <= word_count <= end:
            word_count_lists[word_count].append(data_point)

    print('')

    # Sample until subset size is reached or all data points have been sampled.
    while len(substitutes) < substitutions and len(word_count_lists) > 0:
        empty = []
        keys = list(word_count_lists.keys())

        # Randomise order for each selection round for true uniform sampling.
        random.shuffle(keys)
        for key in keys:
            completion_bar(len(substitutes), substitutions, text='Sampling substitutes')
            if len(substitutes) >= substitutions:
                return substitutes

            if len(word_count_lists[key]) == 0:
                empty.append(key)
                continue

            selected = word_count_lists[key].pop(0)
            if selected[duplicate_column].replace('\n', '') not in uniques:
                substitutes.append(selected)

        # Delete integer value lists where all data points are sampled.
        for key in empty:
            del word_count_lists[key]

    return substitutions

