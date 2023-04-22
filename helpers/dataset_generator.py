import requests
import datasets as ds
import csv
import os
import re
import json
from pathlib import Path


def length_of(string):
    return len(re.findall(r'\w+', string))


def sort_by_count(dataset, column_name):
    """
    Sorts a column of a dataset by the word count, largest to smallest.

    Parameters
    ----------
    dataset : dict
        Dataset to extract texts from
    column_name : str
        Column name containing the texts which are to be extracted.

    Returns
    -------
    list
        A sorted list of dictionaries containing 'title', 'text' and 'word_count'
    """

    new_dataset = []

    # Format data_points and retrieve word_count
    for i, data_point in enumerate(dataset):
        total = len(dataset)
        if i % 10000 == 0:
            print('\r', f'Sorting progress: {round(i/total*100)}%', end="")
        word_count = length_of(data_point[column_name])
        if word_count < 50:
            continue

        new_dataset.append({
            'title': data_point['title'],
            'text': data_point[column_name],
            'word_count': word_count,
        })

    # Sort the list
    sort_criteria = lambda instance: instance['word_count']
    new_dataset.sort(key=sort_criteria, reverse=True)
    print()

    return new_dataset


def generate_abstracts(quantity, data, target_file_name, target_dir_path="./", start=0, iterate_forward=True):
    """Generates scientific abstracts from Open-AI's ChatGPT-API using the GPT-turbo-3.5 model.
        Continuously writes to CSV incase bad API-responses.

        Parameters
        ----------
        quantity : int
            How many samples to generate.
        data : list[dict]
            A list of dictionaries holding a 'title'- and 'word count'-fields to generate an abstract from.
        target_file_name: str
            Name of the csv-file
        target_dir_path: str
            Path to the target directory for the reformatted dataset.
        start : int, optional
            Index of the data-list from which the generation should start from (inclusive).
        iterate_forward : bool, optional
            Iteration-direction when generating samples from data-list.
    """

    # Set up the API key and endpoint
    api_key = Path('../../api-keys/openai_key').read_text()
    url = "https://api.openai.com/v1/chat/completions"

    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Set up the input prompts
    with open('chatgpt-prompt.json') as file:
        prompts = json.load(file)
    system_prompt = prompts['system_instruction']
    user_prompt = prompts['user_base_instruction']

    # Initiate CSV
    path_to_csv = target_dir_path + "/" + target_file_name + ".csv"
    if Path(path_to_csv).is_file():
        print(
            "Security measure: This file already exists. Pick another filename or delete this manually if you wish "
            "to overwrite it.")
        return

    os.makedirs(target_dir_path, exist_ok=True)
    with open(path_to_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'real_abstract', 'real_word_count', 'generated_abstract', 'generated_word_count'])

    # Reverse if set
    if not iterate_forward:
        data.reverse()

    print()
    # Generation loop
    for i in range(start, start+quantity):
        print('\r', f'Generated: {i}/{quantity}', end="")

        # Set title, abstract and word count goal
        title = data[i]['title']
        real_abstract = data[i]['text']
        real_word_count = data[i]['word_count']
        user_prompt += f'Title: {title}\nWord count goal: {real_word_count}'

        # Set up parameters for the API-call
        content = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}
                         ],
        }

        # Make the API call
        response = requests.post(url, headers=headers, json=content)

        if response.status_code == 200:
            # Extract the generated text and its word count
            generated_abstract = response.json()["choices"][0]["message"]['content']
            generated_word_count = length_of(generated_abstract)

            # Write to CSV
            with open(path_to_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [title, real_abstract, real_word_count, generated_abstract, generated_word_count])
        else:
            print("Error:", response.status_code, response.text)
            return  # Quit if something fails to not waste API-usage


def get_models():
    api_key = Path('../../api-keys/openai_key').read_text()
    url = "https://api.openai.com/v1/models"

    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the generated text
        generated_text = response.json()["data"]
        for data_point in generated_text:
            print(data_point)
    else:
        print("Error:", response.status_code, response.text)


dataset = ds.load_dataset("gfissore/arxiv-abstracts-2021")['train']
sorted_dataset = sort_by_count(dataset, 'abstract')

# for i in range(0, -10, -1):
# print(f'\nWord_count: {sorted_dataset[i]["word_count"]}\nTitle: {sorted_dataset[i]["title"]}')

generate_abstracts(quantity=1000,
                   data=sorted_dataset,
                   target_file_name='research_abstracts',
                   target_dir_path='./../../datasets/origins/research-abstracts')
