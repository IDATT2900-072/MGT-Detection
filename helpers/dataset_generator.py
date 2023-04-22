import requests
import datasets as ds
import re
import json
from pathlib import Path


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
    for data_point in dataset:
        word_count = len(re.findall(r'\w+', data_point[column_name]))

        new_dataset.append({
            'title': data_point['title'],
            'text': data_point[column_name],
            'word_count': word_count,
        })

    # Sort the list
    sort_criteria = lambda instance: instance['word_count']
    new_dataset.sort(key=sort_criteria, reverse=True)

    return new_dataset


def generate_abstracts(quantity, abstracts):
    """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        file_loc : str
            The file location of the spreadsheet
        print_cols : bool, optional
            A flag used to print the columns to the console (default is
            False)

        Returns
        -------
        list
            a list of strings used that are the header columns
        """

    # Set up the API key and endpoint
    api_key = "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"
    url = "https://api.openai.com/v1/engines/davinci-codex/completions"

    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Set up the input prompt
    prompt = Path('chatgpt-prompt.txt').read_text()
    print(prompt)

    # Set up parameters for the API call
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 1.0
    }

    # Make the API call
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the generated text
        generated_text = response.json()["choices"][0]["text"]
        print("Generated text:", generated_text)
    else:
        print("Error:", response.status_code, response.text)


dataset = ds.load_dataset("gfissore/arxiv-abstracts-2021")['train']
sorted_dataset = sort_by_count(dataset, 'abstract')

for i in range(0, -10, -1):
    print(f'\nWord_count: {sorted_dataset[i]["word_count"]}\nTitle: {sorted_dataset[i]["title"]}')
