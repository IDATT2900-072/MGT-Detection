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


def plot_distribution(plots: list[dict], start, end, sigma=2, x_label=None, y_label=None, save_to=None, title=None, y_lim=None, h_lines=None, v_lines=None, legend_offset=1.0):
    # Set the plot style
    with plt.style.context('ggplot'):
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.patch.set_facecolor('lightgrey')
        ax.patch.set_alpha(0.3)

        # Set custom y-axis limits if provided
        if y_lim:
            ax.set_ylim(y_lim)

        # Set horizontal lines if provided
        if h_lines:
            for h_line in h_lines:
                ax.axhline(h_line['value'], color=h_line['color'], linestyle='--', alpha=h_line['alpha'])
                ax.text(h_line['offset'][0], h_line['value'] + h_line['offset'][1], h_line['text'], color=h_line['color'])

        # Set vertical lines if provided
        if v_lines:
            for v_line in v_lines:
                ax.axvline(v_line['value'], color=v_line['color'], linestyle='--', alpha=0.8)
                ax.text(v_line['value'] + v_line['offset'][0], v_line['offset'][1], v_line['text'], color=v_line['color'])

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

            if plot['mode']:
                # Find the maximum y-value and its index in the smoothed_counts array
                max_y_index = np.argmax(smoothed_counts)
                max_y_value = smoothed_counts[max_y_index]
                max_x_value = x_values[max_y_index]

                # Draw a dashed line from the maximum y-value to the x-axis
                ax.axvline(max_x_value, ymin=0, ymax=max_y_value / ax.get_ylim()[1], color=plot['color'], linestyle='--', alpha=plot['alpha'])

                # Display the x-value at the base of the dashed line
                ax.text(max_x_value + 2 + 3 * len(str(max_x_value)), 0, f"{max_x_value}", color=plot['color'], ha='center', va='bottom')

        # Set labels and title if provided
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # Add a legend with an offset
        ax.legend(facecolor='white', bbox_to_anchor=(legend_offset, 1))

    # Display the plot
    plt.subplots_adjust(left=0.07, bottom=0.143, right=0.93, top=0.943)

    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_histogram(plots: list[dict], start, end, sigma=2, save_to=None):
    bins = end-start+1

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

            ax.hist(counts, bins=bins, range=(start, end), alpha=plot['alpha'], label=plot['display'], color=plot['color'])

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
