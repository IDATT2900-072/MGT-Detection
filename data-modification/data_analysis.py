import matplotlib
import pandas as pd
import numpy as np
import random

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

matplotlib.use('MacOSX')
from datasets import load_dataset, concatenate_datasets, Dataset
from data_processing import filter_and_count, completion_bar, count_and_reformat, sample_uniform_subset, \
    substitute_duplicates_uniform


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


def plot_distribution(plots: list[dict], start, end, sigma=2, save_to=None, title=None, ylim=None):
    # Set the plot style
    with plt.style.context('ggplot'):
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.patch.set_facecolor('lightgrey')
        ax.patch.set_alpha(0.3)

        for i, plot in enumerate(plots):
            counts = np.zeros(end - start + 1)

            for data_point in plot['dataset']:
                count = data_point[plot['column_name']]

                if start <= count <= end:
                    index = data_point[plot['column_name']] - start
                    counts[index] += 1

            # Apply the Gaussian filter
            smoothed_counts = gaussian_filter1d(counts, sigma)

            # Plot the smoothed data with a label for the legend
            x_values = np.arange(start, end + 1)
            ax.plot(x_values, smoothed_counts, label=plot['display'], alpha=plot['alpha'], color=plot['color'])

        # Set labels and title
        ax.set_xlabel('Length of text in words')
        ax.set_ylabel('Number of texts')
        if title:
            ax.set_title(title)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Add a legend
        ax.legend(facecolor='white')

        # Set custom y-axis limits if provided
        if ylim:
            ax.set_ylim(ylim)

    # Display the plot
    plt.subplots_adjust(left=0.07, bottom=0.143, right=0.93, top=0.943)

    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_histogram(plots: list[dict], start, end, sigma=2, save_to=None, bins=None):
    # Set the plot style
    with plt.style.context('ggplot'):
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.patch.set_facecolor('lightgrey')
        ax.patch.set_alpha(0.3)

        for i, plot in enumerate(plots):
            counts = []

            for data_point in plot['dataset']:
                count = data_point[plot['column_name']]

                if start <= count <= end:
                    counts.append(count)

            # Apply the Gaussian filter if necessary
            if sigma:
                hist_data, bin_edges = np.histogram(counts, bins=bins, range=(start, end))
                smoothed_counts = gaussian_filter1d(hist_data, sigma)
                x_values = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.plot(x_values, smoothed_counts, label=plot['display'], color=plot['color'])
            else:
                ax.hist(counts, bins=bins, range=(start, end), alpha=0.5, label=plot['display'], color=plot['color'])

        # Set labels and title
        ax.set_xlabel('Length of text in words')
        ax.set_ylabel('Number of texts')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Add a legend
        ax.legend(facecolor='white')

    # Adjust the plot margins
    plt.subplots_adjust(left=0.07, bottom=0.143, right=0.93, top=0.943)

    # Save and display the plot
    if save_to:
        plt.savefig(save_to)
    plt.show()

