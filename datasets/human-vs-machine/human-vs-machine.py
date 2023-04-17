import os
import csv
import random

import datasets

_CITATION = """\
@InProceedings{human-vs-machine:dataset,
title = {Human vs Machine dataset collection},
author={Nicolai Sivesind & Andreas Bentzen Winje},
year={2023}
}
"""

_DESCRIPTION = """This dataset contains labeled data with human-produced and LLM-generated texts from various domains: 
Wikipedia introductions and academic articles."""

_HOMEPAGE = ""

_LICENSE = ""


class DomainConfig(datasets.BuilderConfig):
    def __init__(self, name, url, **kwargs):
        super(DomainConfig, self).__init__(name=name, **kwargs)
        self.url = url


class MixedLabeled(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DomainConfig(name="wiki_labeled", url="./wiki-labeled.csv", description="Wikipedia introductions"),
        DomainConfig(name="academic_labeled", url="./academic-labeled.csv", description="Academic articles"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "class label": datasets.ClassLabel(names=["real", "generated"]),
                "text": datasets.Value("string"),
            })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("text", "class label"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(self.config.url)

        data = []
        with open(downloaded_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"class label": int(row["class label"]), "text": row["text"]})

        train_alloc = 0.7
        train_split, test_split = int(train_alloc * len(data)), int((train_alloc + ((1 - train_alloc) / 2)) * len(data))
        train_data, test_data, validation_data = data[:train_split], data[train_split:test_split], data[test_split:]

        random.seed(42)
        random.shuffle(data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": train_data,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": test_data,
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": validation_data,
                    "split": "validation"
                },
            ),
        ]

    def _generate_examples(self, data, split):
        for key, row in enumerate(data):
            yield key, {
                "class label": row["class label"],
                "text": row["text"],
            }
