import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from .data_processing import filter_and_count

matplotlib.use('MacOSX')


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


def plot_distribution(plots: list[dict], start, end, sigma=2, x_label=None, y_label=None, save_to=None, title=None,
                      y_lim=None, h_lines=None, v_lines=None, legend_offset=1.0):
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
                ax.text(h_line['offset'][0], h_line['value'] + h_line['offset'][1], h_line['text'],
                        color=h_line['color'])

        # Set vertical lines if provided
        if v_lines:
            for v_line in v_lines:
                ax.axvline(v_line['value'], color=v_line['color'], linestyle='--', alpha=0.8)
                ax.text(v_line['value'] + v_line['offset'][0], v_line['offset'][1], v_line['text'],
                        color=v_line['color'])

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
                ax.axvline(max_x_value, ymin=0, ymax=max_y_value / ax.get_ylim()[1], color=plot['color'],
                           linestyle='--', alpha=plot['alpha'])

                # Display the x-value at the base of the dashed line
                ax.text(max_x_value + 2 + 3 * len(str(max_x_value)), 0, f"{max_x_value}", color=plot['color'],
                        ha='center', va='bottom')

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


def plot_histogram(plots: list[dict], start, end, x_label, y_label, y_lim=None, save_to=None):
    bins = end - start + 1

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

            ax.hist(counts, bins=bins, range=(start, end), alpha=plot['alpha'], label=plot['display'],
                    color=plot['color'])

        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if y_lim:
            ax.set_ylim(y_lim)

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


def calculate_mean_y_per_x(x_values, y_values):
    xy_dict = {}
    for x, y in zip(x_values, y_values):
        if x not in xy_dict:
            xy_dict[x] = {'sum': 0, 'count': 0}
        xy_dict[x]['sum'] += y
        xy_dict[x]['count'] += 1

    mean_y_values = {x: y_data['sum'] / y_data['count'] for x, y_data in xy_dict.items()}
    return list(mean_y_values.keys()), list(mean_y_values.values())


def plot_scatter(plots: list[dict], d_lines=None, h_lines=None, v_lines=None, x_label=None, y_label=None, y_lim=None,
                 legend_offset=(1.0, 1.0), average_curve=None, sigma=2, correlations=None):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.patch.set_facecolor('lightgrey')
        ax.patch.set_alpha(0.3)

        if y_lim:
            ax.set_ylim(y_lim)

        if h_lines:
            for h_line in h_lines:
                ax.axhline(h_line['value'], color=h_line['color'], linestyle='--', alpha=h_line['alpha'])
                ax.text(h_line['offset'][0], h_line['value'] + h_line['offset'][1], h_line['text'],
                        color=h_line['color'])

        if v_lines:
            for v_line in v_lines:
                ax.axvline(v_line['value'], color=v_line['color'], linestyle='--', alpha=0.8)
                ax.text(v_line['value'] + v_line['offset'][0], v_line['offset'][1], v_line['text'],
                        color=v_line['color'])

        for plot in plots:
            x_values = [data_point[plot['x']] for data_point in plot['dataset']]
            y_values = [data_point[plot['y']] for data_point in plot['dataset']]

            ax.scatter(x_values, y_values, label=plot['display'], alpha=plot['alpha'], color=plot['color'])

            if average_curve:
                x_unique, y_mean = calculate_mean_y_per_x(x_values, y_values)
                x_unique, y_mean = zip(*sorted(zip(x_unique, y_mean)))
                y_mean_filtered = gaussian_filter1d(y_mean, sigma)
                ax.plot(x_unique, y_mean_filtered, label=average_curve['display'], color=average_curve['color'],
                        alpha=average_curve['alpha'])

            if correlations:
                for correlation in correlations:
                    interval = correlation.get('interval', (min(x_values), max(x_values)))
                    x_interval_values = [x for x in x_values if interval[0] <= x <= interval[1]]
                    y_interval_values = [y for x, y in zip(x_values, y_values) if interval[0] <= x <= interval[1]]

                    interval_text = f"∀x ∈ [{interval[0]}, {interval[1]}]:"

                    corr, p_value = pearsonr(x_interval_values, y_interval_values)
                    ax.text(correlation['positioning'][0],
                            correlation['positioning'][1],
                            f"{interval_text}{' ' * correlation['spaces'][0]} n = {len(x_interval_values)},"
                            f"{' ' * correlation['spaces'][1]} r = {corr:.2f}{' ' * correlation['spaces'][2]}∧"
                            f"{' ' * correlation['spaces'][2]} p={p_value:.2f}",
                            color=correlation['color'],
                            alpha=correlation['alpha'], bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

        if d_lines:
            for d_line in d_lines:
                x_start, y_start = d_line['start']
                x_increment, y_increment = d_line['increment']
                x_max = max(ax.get_xlim())
                y_max = max(ax.get_ylim())

                x_end = min(x_max, (y_max - y_start) / y_increment * x_increment + x_start)
                y_end = x_end * y_increment / x_increment + y_start - x_start * y_increment / x_increment

                ax.plot([x_start, x_end], [y_start, y_end], label=d_line['display'], color=d_line['color'],
                        linestyle='--', alpha=d_line['alpha'])

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    ax.legend(facecolor='white', bbox_to_anchor=(legend_offset[0], legend_offset[1]))
    plt.show()


def plot_loss_curves(plots, deviations=None, x_label=None, y_label=None, v_lines=None, legend_offset=(1.0, 1.0), sigma=2):
    for plot in plots:
        dataset = plot['dataset']
        positive_loss = []
        negative_loss = []
        zero_loss = 0

        for data_point in dataset:
            real_word_count = data_point[plot['benchmark']]
            generated_word_count = data_point[plot['predicted']]
            loss = generated_word_count - real_word_count

            if loss > 0:
                positive_loss.append((real_word_count, abs(loss)))
            elif loss < 0:
                negative_loss.append((real_word_count, abs(loss)))
            else:
                zero_loss += 1

        with plt.style.context('ggplot'):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.patch.set_facecolor('lightgrey')
            ax.patch.set_alpha(0.3)

            if v_lines:
                for v_line in v_lines:
                    ax.axvline(v_line['value'], color=v_line['color'], linestyle='--', alpha=0.8)
                    ax.text(v_line['value'] + v_line['offset'][0], v_line['offset'][1], v_line['text'],
                            color=v_line['color'])

            all_losses = positive_loss + negative_loss

            if all_losses:
                x_values, y_values = zip(*all_losses)
                x_unique, y_mean_abs_dev = calculate_mean_y_per_x(x_values, y_values)
                x_unique, y_mean_abs_dev = zip(*sorted(zip(x_unique, y_mean_abs_dev)))
                y_mean_abs_dev_filtered = gaussian_filter1d(y_mean_abs_dev, sigma)
                ax.plot(x_unique, y_mean_abs_dev_filtered, label=plot['mean-abs-display'], color=plot['mean-abs-color'],
                        alpha=plot['alpha'])

            if positive_loss:
                x_values, y_values = zip(*positive_loss)
                x_unique, y_mean = calculate_mean_y_per_x(x_values, y_values)
                x_unique, y_mean = zip(*sorted(zip(x_unique, y_mean)))
                y_mean_filtered = gaussian_filter1d(y_mean, sigma)
                ax.plot(x_unique, y_mean_filtered, label=plot['positive-display'], color=plot['positive-color'],
                        alpha=plot['alpha'])

            if negative_loss:
                x_values, y_values = zip(*negative_loss)
                x_unique, y_mean = calculate_mean_y_per_x(x_values, y_values)
                x_unique, y_mean = zip(*sorted(zip(x_unique, y_mean)))
                y_mean_filtered = gaussian_filter1d(y_mean, sigma)
                ax.plot(x_unique, y_mean_filtered, label=plot['negative-display'], color=plot['negative-color'],
                        alpha=plot['alpha'])

            if deviations:
                for deviation in deviations:
                    if 'zero-text' in deviation:
                        ax.text(deviation['positioning'][0], deviation['positioning'][1],
                                f"{deviation['zero-text']} {zero_loss}", color=deviation['color'],
                                alpha=deviation['alpha'])
                    if 'positive-text' in deviation:
                        ax.text(deviation['positioning'][0], deviation['positioning'][1],
                                f"{deviation['positive-text']} {len(positive_loss)}", color=deviation['color'],
                                alpha=deviation['alpha'])
                    if 'negative-text' in deviation:
                        ax.text(deviation['positioning'][0], deviation['positioning'][1],
                                f"{deviation['negative-text']} {len(negative_loss)}", color=deviation['color'],
                                alpha=deviation['alpha'])

            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)

            plt.subplots_adjust(left=0.07, bottom=0.143, right=0.93, top=0.943)
            ax.legend(facecolor='white', bbox_to_anchor=(legend_offset[0], legend_offset[1]))
            plt.show()
