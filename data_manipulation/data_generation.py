import numpy as np
import random

import requests
import csv
import os
import json
from pathlib import Path
import time

from .data_processing import word_length_of, filter_and_count, completion_bar
from .csv_writing import create_csv_if_nonexistent, write_csv_row, path_to_csv

# Constants
API_KEY = Path('../../api-keys/openai_key').read_text()
GPT_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo-0301"


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
    with open('../prompts/generation-prompts.json') as file:
        prompts = json.load(file)
    system_prompt = prompts['system_instruction']
    user_base_prompt = prompts['user_base_instruction']
    expand_base_prompt = prompts['user_expand_instruction']

    # Initiate CSV
    columns = ['title', 'real_abstract', 'real_word_count', 'generated_abstract', 'generated_word_count']
    create_csv_if_nonexistent(columns, target_dir_path, target_file_name)

    # Reverse data-list if set
    if not iterate_forward:
        data.reverse()

    # Generation loop
    data_length = len(data)
    for i in range(start_index, data_length):
        print('\r', f'Generating: {i + 1}/{data_length}', end="")

        # Set title, abstract and word count goal
        title = data[i]['title'].replace('\n', '')
        real_abstract = data[i]['abstract']
        real_word_count = data[i]['word_count']
        user_prompt = user_base_prompt.format(title=title, word_count_goal=real_word_count)

        # If debugging, skips API-calls
        if debug:
            continue

        # Generate abstract
        generated_abstract, generated_word_count = generate_GPT_abstract(system_prompt, user_prompt)

        # Write to CSV
        row = [title, real_abstract, real_word_count, generated_abstract, generated_word_count]
        write_csv_row(row, path_to_csv(target_dir_path, target_file_name))

    print("\nAbstract generation complete.\n\n")


def generate_GPT_abstract(system_prompt, user_prompt, attempts=0):
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
        generated_word_count = word_length_of(generated_abstract)
        return generated_abstract, generated_word_count
    else:
        if attempts < 5:
            print("\n API-error. Reattempting API-call")
            time.sleep(5)
            return generate_GPT_abstract(system_prompt, user_prompt, attempts + 1)
        else:
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
            print(model_specs, "\n")
    else:
        print("Error:", response.status_code, response.text)
