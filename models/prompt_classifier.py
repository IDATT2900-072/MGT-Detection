import requests
import csv
import os
import json
import time
import torch

from pathlib import Path
from data_manipulation.data_processing import sample_uniform_subset
from data_manipulation.csv_writing import create_csv_if_nonexistent, write_csv_row, path_to_csv


def init_csv(target_dir_path, target_file_name):
    # Initiate CSV
    path = target_dir_path + "/" + target_file_name + ".csv"
    if not Path(path).is_file():
        print("No file already exists. Creating blank CSV\n")
        os.makedirs(target_dir_path, exist_ok=True)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['title', 'real_abstract', 'real_word_count', 'generated_abstract', 'generated_word_count'])
    else:
        print("CSV-file already exists. Will append new rows to existing document. Cancel execution if this is not "
              "intended.\n")


class Prompter:
    with open('../prompts/classification-prompts.json') as file:
        prompts = json.load(file)

    def __init__(self, dataset, api_key, model):
        self.dataset = dataset
        self.API_KEY = api_key
        self.GPT_API_URL = "https://api.openai.com/v1/completions"
        self.MODEL = model

    def classify_set(self, target_dir_path, target_files_base_name, num_classifications, title_column, human_column,
                     generated_column, human_word_count_column, generated_word_count_column, min_word_count,
                     max_word_count, zero_shot=True, few_shot=True):
        # Sample a wc-uniform set of prompt-task-bundles
        task_bundles = self.sample_few_shot_bundles(n_bundles=num_classifications,
                                                    title_column=title_column,
                                                    human_column=human_column,
                                                    generated_column=generated_column,
                                                    human_word_count_column=human_word_count_column,
                                                    generated_word_count_column=generated_word_count_column,
                                                    min_word_count=min_word_count,
                                                    max_word_count=max_word_count)

        # Create CSV
        columns = ['title', 'word_count', 'label',
                   'predicted', 'human_probability', 'generated_probability', 'prompt_response']
        zero_shot_file_name = target_files_base_name + "_zero-shot"
        few_shot_file_name = target_files_base_name + "_few-shot"

        if zero_shot:
            create_csv_if_nonexistent(columns, target_dir_path, zero_shot_file_name)
        if few_shot:
            create_csv_if_nonexistent(columns, target_dir_path, few_shot_file_name)

        # Perform classification
        bundle_size = len(task_bundles)
        for i, bundle in enumerate(task_bundles):
            # Zero-shot
            if zero_shot:
                print('\r', f'Zero-shot: {i + 1}/{bundle_size}', end="")
                prediction, human_probability, generated_probability, prompt_response = self.zero_shot(bundle['input_text'])
                row = [bundle['input_title'], bundle['input_word_count'], bundle['input_label'],
                       prediction, human_probability, generated_probability, prompt_response]
                write_csv_row(row, path_to_csv(target_files_base_name, zero_shot_file_name))

            # Few-shot
            if few_shot:
                print('\r', f'Few-shot: {i + 1}/{bundle_size}', end="")
                prediction, human_probability, generated_probability, prompt_response = self.few_shot(
                    bundle['input_text'])
                row = [bundle['input_title'], bundle['input_word_count'], bundle['input_label'],
                       prediction, human_probability, generated_probability, prompt_response]
                write_csv_row(row, path_to_csv(target_files_base_name, zero_shot_file_name))

        print("\n\nClassification complete")

    def sample_few_shot_bundles(self, n_bundles, title_column, human_column, generated_column, human_word_count_column,
                                generated_word_count_column, min_word_count, max_word_count):
        subset_size = n_bundles * 4  # 6 examples, 2 from each row (3 rows), 1 input_text from 1 row.
        print(subset_size)

        subset = sample_uniform_subset(self.dataset, human_word_count_column, subset_size, min_word_count,
                                       max_word_count)
        print(len(subset))
        subset.sort(key=lambda x: x[human_word_count_column])

        task_bundles = []
        for i in range(0, len(subset), 8):
            human_text = {
                "example_1": subset[i][human_column],
                "example_2": subset[i][generated_column],
                "example_3": subset[i + 1][human_column],
                "example_4": subset[i + 1][generated_column],
                "example_5": subset[i + 2][human_column],
                "example_6": subset[i + 2][generated_column],
                "input_text": subset[i + 3][human_column],
                "input_word_count": subset[i + 3][human_word_count_column],
                "input_title": subset[i + 3][title_column],
                "input_label": "Human"
            }

            generated_text = {
                "example_1": subset[i + 4][human_column],
                "example_2": subset[i + 4][generated_column],
                "example_3": subset[i + 5][human_column],
                "example_4": subset[i + 5][generated_column],
                "example_5": subset[i + 6][human_column],
                "example_6": subset[i + 6][generated_column],
                "input_text": subset[i + 7][generated_column],
                "input_word_count": subset[i + 7][generated_word_count_column],
                "input_title": subset[i + 7][title_column],
                "input_label": "Generated"
            }

            task_bundles.append(human_text)
            task_bundles.append(generated_text)

        return task_bundles

    def zero_shot(self, text_sample):
        prompt = self.prompts["zero-shot"].format(input_text=text_sample)
        return self.classify(prompt)

    def few_shot(self, task_bundle):
        prompt = self.prompts["few-shot"].format(human_text_1=task_bundle['example_1'],
                                                 generated_text_1=task_bundle['example_2'],
                                                 human_text_2=task_bundle['example_3'],
                                                 generated_text_2=task_bundle['example_4'],
                                                 human_text_3=task_bundle['example_5'],
                                                 generated_text_3=task_bundle['example_6'],
                                                 input_text=task_bundle['input_text'])
        return self.classify(prompt)

    def classify(self, prompt, attempts=0):
        # Set up content for the API-call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }

        content = {
            "model": self.MODEL,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0.3,
            "top_p": 1,
            "logprobs": 5,
            "logit_bias": {
                "20490": 20,  # "Human
                "8645": 20,  # "Gener"
                "515": 20,  # "ated"
            },
            "n": 1
        }

        # Make the API call
        response = requests.post(self.GPT_API_URL, headers=headers, json=content)

        if response.status_code == 200:
            response = response.json()

            answer = response["choices"][0]["text"].strip()

            # Extract the logits for human-produced and machine-generated labels
            logits_human = response["choices"][0]["logprobs"]["top_logprobs"][0]["Human"]
            logits_generated = response["choices"][0]["logprobs"]["top_logprobs"][0]["Gener"]

            # Calculate probabilities using softmax function
            logits = torch.tensor([logits_human, logits_generated])
            probabilities = torch.softmax(logits, dim=-1)
            human_probability = probabilities[0].item()
            machine_probability = probabilities[1].item()

            prediction = "human" if human_probability > machine_probability else "generated"

            return prediction, human_probability, machine_probability, answer
        else:
            if attempts < 3:
                print(response.text)
                print(f"\n API-error. Reattempting API-call. Attempt {attempts + 1}")
                time.sleep(5)
                return self.classify(prompt, attempts + 1)
            else:
                raise RuntimeError(f"API-error: {response.status_code}, {response.text}")
