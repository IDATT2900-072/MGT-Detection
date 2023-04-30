import matplotlib
import pandas as pd
import numpy as np
import random

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

matplotlib.use('MacOSX')
from datasets import load_dataset, concatenate_datasets
from data_processing import filter_and_count, completion_bar, count_and_reformat, sample_uniform_subset


def display_word_count_intervals(dataset, intervals, column_name):
    # Count word ranges
    for interval in intervals:
        _, interval['num_matches'] = filter_and_count(dataset, column_name, interval['min'], interval['max'])

    # Sort intervals by the 'min' value
    intervals.sort(key=lambda x: x['min'])

    # Extract values for the x and y axes
    x_labels = [f"{interval['min']}-{interval['max']}" for interval in intervals]
    y_values = [interval['num_matches'] for interval in intervals]

    # Display the plot
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        bar_container = ax.bar(x_labels, y_values)
        ax.bar_label(bar_container, padding=3)

        plt.xlabel('Word count interval')
        plt.ylabel('Number of data points')
        plt.title('Word count distribution')

        plt.show()


def plot_gausian_filter(dataset, columns, start, end, sigma=2):
    # Set the plot style
    with plt.style.context('ggplot'):
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, column in enumerate(columns):
            counts = np.zeros(end - start + 1)
            for data_point in dataset:
                index = data_point[column['column_name']] - start
                counts[index] += 1

            # Apply the Gaussian filter
            smoothed_counts = gaussian_filter1d(counts, sigma)

            # Plot the smoothed data with a label for the legend
            x_values = np.arange(start, end + 1)
            ax.plot(x_values, smoothed_counts, label=column['display'])

        # Set labels and title
        ax.set_xlabel('Length of text in words')
        ax.set_ylabel('Number of texts')
        ax.set_title('Word count distribution')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Add a legend
        ax.legend()

    # Display the plot
    plt.show()


# Execution
# dataset_10k = load_dataset('csv', data_files='../../datasets/origins/research-abstracts/research_abstracts.csv')[
#     'train']
# dataset_2m = count_and_reformat(dataset=load_dataset("gfissore/arxiv-abstracts-2021")['train'],
#                                 count_column='abstract',
#                                 retain_columns=['title', 'abstract'])
#
# wc_uniform = sample_uniform_subset(dataset=dataset_2m,
#                                    column='word_count',
#                                    start=50,
#                                    end=600,
#                                    subset_size=10000)
#
# wc_random = sample_random_subset(dataset=dataset_2m,
#                                  column_name='word_count',
#                                  start=50,
#                                  end=600,
#                                  subset_size=10000)

dataset_1 = \
    load_dataset('csv', data_files='../../datasets/origins/research-abstracts/research_abstracts-uniform-old.csv')[
        'train']
dataset_2 = \
    load_dataset('csv', data_files='../../datasets/origins/research-abstracts/research_abstracts-uniform.csv')['train']

dataset = concatenate_datasets([dataset_1, dataset_2])
unique = dataset.unique('title')

print('list: ', dataset.num_rows)
print('unique: ', len(unique))
plot_gausian_filter(dataset=dataset,
                    columns=[{'column_name': 'real_word_count', 'display': 'Generated texts'},
                             {'column_name': 'generated_word_count', 'display': 'Real texts'}],
                    start=50,
                    end=600)
