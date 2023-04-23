import random

import requests
import datasets as ds
import csv
import os
import re
import json
from pathlib import Path

# Constants
API_KEY = Path('../../api-keys/openai_key').read_text()
GPT_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo-0301"


def length_of(string):
    return len(re.findall(r'\w+', string))


def count_and_reformat(dataset, column_name):
    """
    Counts text length in words for every data point in a dataset column.

    Parameters
    ----------
    dataset : dict
        Dataset to extract texts from
    column_name : str
        Column name containing the texts which are to be extracted.

    Returns
    -------
    list
        A list of dictionaries containing 'title', 'text' and 'word_count'
    """

    new_dataset = []

    # Format data_points and retrieve word_count
    for i, data_point in enumerate(dataset):
        total = len(dataset)
        if i % int(total / 100) == 0:
            print('\r', f'Counting words: {round(i / total * 100)}%', end="")
        word_count = length_of(data_point[column_name])
        if word_count < 50:
            continue

        new_dataset.append({
            'title': data_point['title'],
            'text': data_point[column_name],
            'word_count': word_count,
        })

    return new_dataset


def filter_list(data, word_count_min, word_count_max, quantity):
    """
    Filters the list, removing entire with a word_count outside the specified rang, and randomly selects the desired
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
        A list with the length of 'quantity' containing the filtered elements from the original data-list. The elements
        are randomly selected from the filtered domain and a sorted in descending order based on their word_count value.
    """
    if word_count_min > word_count_max:
        raise Exception("word_count_min cannot be larger than word_count_max.")
    # Filter and select
    filtered_dataset = list(filter(lambda instance: word_count_min <= instance['word_count'] <= word_count_max, data))
    filter_domain_size = len(filtered_dataset)
    print(f'\nFound {filter_domain_size} elements matching the filter.')

    if filter_domain_size < quantity:
        print(f"Number of filtered elements ({filter_domain_size}), is less than desired quantity ({quantity}).")
        quantity = filter_domain_size

    # Select random elements
    random.seed(42)
    filtered_dataset = random.choices(filtered_dataset, k=quantity)
    print(f'Returned list is of length {len(filtered_dataset)}.')

    # Sort in descending manner
    filtered_dataset.sort(key=lambda instance: instance['word_count'], reverse=True)
    return filtered_dataset


def generate_abstracts(data, target_file_name, target_dir_path="./", start_index=0, iterate_forward=True, debug=False):
    """
    Generates scientific abstracts from Open-AI's ChatGPT-API using the GPT-turbo-3.5 model.
    If file already exists, generated abstracts will be appended to the file. The previous content of the file will not
    be removed unless done manually. This is a precaution to prevent unintended loss of previous generations.
    Continuously writes to CSV incase bad API-responses.

    Parameters
    ----------
    data : list[dict]
        A list of dictionaries holding a 'title'- and 'word count'-fields to generate an abstract from.
    target_file_name: str
        Name of the csv-file
    target_dir_path: str
        Path to the target directory for the reformatted dataset.
    start_index : int, optional
        Index of the data-list from which the generation should start from (inclusive).
    iterate_forward : bool, optional
        Iteration-direction when generating samples from data-list.
    debug : bool, optional
        If set to True, API-calls are skipped.
    """

    # Set up the input prompts
    with open('chatgpt-prompt.json') as file:
        prompts = json.load(file)
    system_prompt = prompts['system_instruction']
    user_base_prompt = prompts['user_base_instruction']
    expand_base_prompt = prompts['user_expand_instruction']

    # Initiate CSV
    path_to_csv = target_dir_path + "/" + target_file_name + ".csv"
    if not Path(path_to_csv).is_file():
        print("No file already exists. Creating blank CSV\n")
        os.makedirs(target_dir_path, exist_ok=True)
        with open(path_to_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['title', 'real_abstract', 'real_word_count', 'generated_abstract', 'generated_word_count'])
    else:
        print("CSV-file already exists. Will append new rows to existing document. Cancel execution if this is not "
              "intended.\n")

        # Reverse data-list if set
    if not iterate_forward:
        data.reverse()

    # Generation loop
    data_length = len(data)
    for i in range(start_index, data_length):
        print('\r', f'Generating: {i + 1}/{data_length - start_index}', end="")

        # Set title, abstract and word count goal
        title = data[i]['title']
        real_abstract = data[i]['text']
        real_word_count = data[i]['word_count']
        user_prompt = user_base_prompt.format(title=title, word_count_goal=real_word_count)

        # If debugging, skips API-calls
        if debug:
            continue

        # Generate abstract
        generated_abstract, generated_word_count = generate_GPT_abstract(system_prompt, user_prompt)

        # Write to CSV
        with open(path_to_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [title, real_abstract, real_word_count, generated_abstract, generated_word_count])

    print("Abstract generation complete.")


def generate_GPT_abstract(system_prompt, user_prompt):
    # Set up content for the API-call
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    content = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
    }

    # Make the API call
    response = requests.post(GPT_API_URL, headers=headers, json=content)

    if response.status_code == 200:
        # Extract the generated text and its word count
        generated_abstract = response.json()["choices"][0]["message"]['content']
        generated_word_count = length_of(generated_abstract)
        return generated_abstract, generated_word_count
    else:
        # Quit if something fails to not waste API-usage
        raise RuntimeError(f"API-error: {response.status_code}, {response.text}")


def get_models():
    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.get("https://api.openai.com/v1/models", headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the generated text
        response_dict = response.json()["data"]
        for model_specs in response_dict:
            print(model_specs)
    else:
        print("Error:", response.status_code, response.text)
