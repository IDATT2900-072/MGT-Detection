import matplotlib.pyplot as plt
from transformers import dataset
from helpers import filter_and_count


def display_word_count_intervals(dataset, intervals):
    # Count word ranges
    for interval in intervals:
        _, interval['num_matches'] = filter_and_count(dataset, interval['min'], interval['max'])

    # Sort intervals by the 'min' value
    intervals.sort(key=lambda x: x['min'])

    # Extract values for the x and y axes
    x_labels = [f"{interval['min']}-{interval['max']}" for interval in intervals]
    y_values = [interval['num_matches'] for interval in intervals]

    # Create the bar plot
    plt.bar(x_labels, y_values)
    plt.xlabel('Word count intervals')
    plt.ylabel('Number of matches')
    plt.title('Word count intervals distribution')
    plt.show()


# Execution

