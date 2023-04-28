import re


def word_length_of(string):
    return len(re.findall(r'\w+', string))


def filter_and_count(dataset, word_count_min, word_count_max):
    if word_count_min > word_count_max:
        raise Exception("word_count_min cannot be larger than word_count_max.")
        # Filter and select
    filtered_dataset = list(filter(lambda instance: word_count_min <= instance['word_count'] <= word_count_max, dataset))
    filter_domain_size = len(filtered_dataset)

    return filtered_dataset, filter_domain_size
