import os
import csv
import random

import datasets

_CITATION = """\
@InProceedings{mixed-labeled:dataset,
title = {Mixed-labeled dataset},
author={Nicolai Sivesind & Andreas Bentzen Winje},
year={2023}
}
"""

_DESCRIPTION = """This dataset contains labeled data with real and generated texts from various domains: 
Wikipedia introductions and academic articles."""

_HOMEPAGE = ""

_LICENSE = ""


class DomainConfig(datasets.BuilderConfig):
    def __init__(self, name, filename, **kwargs):
        super(DomainConfig, self).__init__(name=name, **kwargs)
        self.filename = filename


class MixedLabeled(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DomainConfig(name="wiki_labeled", filename="wiki-labeled.csv", description="Wikipedia introductions"),
        DomainConfig(name="academic_labeled", filename="academic-labeled.csv", description="Academic articles"),
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
        data_dir = os.path.join(os.getcwd(), "dataset")
        filepath = os.path.join(data_dir, self.config.filename)

        data = []
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"class label": int(row["class label"]), "text": row["text"]})

        random.shuffle(data)
        train_split = int(0.8 * len(data))
        train_data, test_data = data[:train_split], data[train_split:]

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
        ]

    def _generate_examples(self, data, split):
        for key, row in enumerate(data):
            yield key, {
                "class label": row["class label"],
                "text": row["text"],
            }
