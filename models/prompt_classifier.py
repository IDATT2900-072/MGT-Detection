from collections import defaultdict

import numpy as np
import requests
import json
import torch
import time
import tiktoken

from data_manipulation.data_processing import sample_uniform_subset
from data_manipulation.csv_writing import create_csv_if_nonexistent, write_csv_row, path_to_csv


class PromptClassifier:
    with open('../prompts/classification-prompts.json') as file:
        prompts = json.load(file)

    def __init__(self, dataset, api_key, model, ban_bias=-100, boost_bias=15):
        self.dataset = dataset
        self.API_KEY = api_key
        self.GPT_API_URL = "https://api.openai.com/v1/completions"
        self.MODEL = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.logit_biases = {}

        # Logit bias
        banned_words = [" ", "\\", "n", "\n", "\n", "Class", " Class", "class", ' class', "Label", "Answer",
                        "Prediction"]
        boosted_words = ["Human", "AI"]

        for banned, boosted in zip(banned_words, boosted_words):
            for token in self.tokenizer.encode(banned):
                self.logit_biases[token] = ban_bias

            for token in self.tokenizer.encode(boosted):
                self.logit_biases[token] = boost_bias

    def classify_set(self, target_dir_path, target_files_base_name, num_classifications, zero_shot_prompt,
                     few_shot_prompt, title_column, human_column, generated_column, human_word_count_column,
                     generated_word_count_column, min_word_count, max_word_count, zero_shot=True, few_shot=True,
                     start_index=0, debug=False):
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
        for i in range(start_index, bundle_size):
            bundle = task_bundles[i]

            # Zero-shot
            if zero_shot:
                print('\r', f'{i + 1}/{bundle_size}, Zero-shot | {i}/{bundle_size}, Few-shot', end="")
                prediction, human_probability, generated_probability, prompt_response = self.zero_shot(bundle['input_text'],
                                                                                                       zero_shot_prompt,
                                                                                                       debug)
                row = [bundle['input_title'], bundle['input_word_count'], bundle['input_label'],
                       prediction, human_probability, generated_probability, prompt_response]

                if debug:
                    print(f"\nResponse: {row}")

                write_csv_row(row, path_to_csv(target_dir_path, zero_shot_file_name))
                time.sleep(2)

            # Few-shot
            if few_shot:
                print('\r', f'{i + 1}/{bundle_size}, Zero-shot | {i + 1}/{bundle_size}, Few-shot', end="")
                prediction, human_probability, generated_probability, prompt_response = self.few_shot(bundle,
                                                                                                      few_shot_prompt,
                                                                                                      debug)
                row = [bundle['input_title'], bundle['input_word_count'], bundle['input_label'],
                       prediction, human_probability, generated_probability, prompt_response]

                if debug:
                    print(f"\nResponse: {row}")

                write_csv_row(row, path_to_csv(target_dir_path, few_shot_file_name))
                time.sleep(2)

            if debug:
                print(f"\n\nDebug. i:{i}")
                return

        print("\n\nClassification complete")

    def sample_few_shot_bundles(self, n_bundles, title_column, human_column, generated_column, human_word_count_column,
                                generated_word_count_column, min_word_count, max_word_count):
        subset_size = n_bundles * 4  # 6 examples, 2 from each row (3 rows), 1 input_text from 1 row.

        subset = sample_uniform_subset(self.dataset, human_word_count_column, subset_size, min_word_count,
                                       max_word_count)
        print(f"\nTotal rows in few-shot task bundles {len(subset)}")

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
                "input_label": "AI"
            }

            task_bundles.append(human_text)
            task_bundles.append(generated_text)

        return task_bundles

    def zero_shot(self, text_sample, instruction_text_name, debug=False):
        prompt = self.prompts[instruction_text_name].format(classification_text=text_sample)

        if debug:
            print(f"\n\n{prompt}")

        return self.classify(prompt, debug=debug)

    def few_shot(self, task_bundle, instruction_text_name, debug=False):
        prompt = self.prompts[instruction_text_name].format(human_text_1=task_bundle['example_1'],
                                                            generated_text_1=task_bundle['example_2'],
                                                            human_text_2=task_bundle['example_3'],
                                                            generated_text_2=task_bundle['example_4'],
                                                            human_text_3=task_bundle['example_5'],
                                                            generated_text_3=task_bundle['example_6'],
                                                            classification_text=task_bundle['input_text'])
        if debug:
            print(f"\n\n{prompt}")

        return self.classify(prompt, debug=debug)

    def classify(self, input_text, debug=False, attempts=0):
        # Set up content for the API-call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }

        hyperparams = {
            "model": self.MODEL,
            "prompt": input_text,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 5,
            "logit_bias": self.logit_biases,
            "n": 1
        }

        # Make the API call
        response = requests.post(self.GPT_API_URL, headers=headers, json=hyperparams)

        if response.status_code == 200:
            response = response.json()

            answer = response["choices"][0]["text"]

            # Retrieve log-probabilities
            top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

            if debug:
                print(f"Top logprobs: {top_logprobs}")
                print(f"Answer:\"{answer}\"")

            logits_human = top_logprobs['Human']  # + top_logprobs[' AI']
            logits_generated = top_logprobs['AI']  # + top_logprobs[' AI']

            # Calculate probabilities using softmax function
            logits = torch.tensor([logits_human, logits_generated])
            probabilities = torch.softmax(logits, dim=-1)
            human_probability = round(probabilities[0].item(), 4)
            machine_probability = round(probabilities[1].item(), 4)

            if debug:
                print(f"Probabilities. {probabilities}")

            prediction = "Human" if human_probability >= machine_probability else "AI"

            return prediction, human_probability, machine_probability, answer
        else:
            if attempts < 3:
                # print(response.text)
                print(f"\n API-error. Reattempting API-call. Attempt {attempts + 1}")
                time.sleep(5)
                return self.classify(input_text, debug=debug, attempts=attempts + 1, )
            else:
                raise RuntimeError(f"API-error: {response.status_code}, {response.text}")
