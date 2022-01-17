# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A thorough walkthrough on how to implement a dataset can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

There are 3 key elements to implementing a dataset:

(1) `_info`: Create a skeletal structure that describes what is in the dataset and the nature of the features.

(2) `_split_generators`: Download and extract data for each split of the data (ex: train/dev/test)

(3) `_generate_examples`: From downloaded + extracted data, process files for the data in a feature format specified in "info".

----------------------
Step 1: Declare imports
Your imports go here; the only mandatory one is `datasets`, as methods and attributes from this library will be used throughout the script.
"""
import json

#  < Your imports here >
import os  # useful for paths

import datasets


"""
Step 2: Create keyword descriptors for your dataset

The following variables are used to populate the dataset entry. Common ones include:

- `_DATASETNAME` = "your_dataset_name"
- `_CITATION`: Latex-style citation of the dataset
- `_DESCRIPTION`: Explanation of the dataset
- `_HOMEPAGE`: Where to find the dataset's hosted location
- `_LICENSE`: License to use the dataset
- `_URLs`: How to download the dataset(s), by name; make this in the form of a dictionary where <dataset_name> is the key and <url_of_dataset> is the balue
- `_VERSION`: Version of the dataset
"""

logger = datasets.logging.get_logger(__name__)


_DATASETNAME = "mashqa"

_CITATION = """\
@inproceedings{zhu-etal-2020-question,
    title = "Question Answering with Long Multiple-Span Answers",
    author = "Zhu, Ming  and
      Ahuja, Aman  and
      Juan, Da-Cheng  and
      Wei, Wei  and
      Reddy, Chandan K.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.342",
    doi = "10.18653/v1/2020.findings-emnlp.342",
    pages = "3840--3849",
    abstract = "Answering questions in many real-world applications often requires complex and precise information excerpted from texts spanned across a long document. However, currently no such annotated dataset is publicly available, which hinders the development of neural question-answering (QA) systems. To this end, we present MASH-QA, a Multiple Answer Spans Healthcare Question Answering dataset from the consumer health domain, where answers may need to be excerpted from multiple, non-consecutive parts of text spanned across a long document. We also propose MultiCo, a neural architecture that is able to capture the relevance among multiple answer spans, by using a query-based contextualized sentence selection approach, for forming the answer to the given question. We also demonstrate that conventional QA models are not suitable for this type of task and perform poorly in this setting. Extensive experiments are conducted, and the experimental results confirm the proposed model significantly outperforms the state-of-the-art QA models in this multi-span QA setting.",
}
"""

_DESCRIPTION = """\
 Multiple Answer Spans Healthcare Question Answering dataset from the consumer health domain, where answers may need to be excerpted from multiple, non-consecutive parts of text spanned across a long document.
"""

_HOMEPAGE = "https://github.com/mingzhu0527/MASHQA"

_LICENSE = "Apache License 2.0"

_URLs = {"mashqa": "https://drive.google.com/file/d/1YJ5Pw7CoBcwKv2YYCguqLF2HQDcPMd8G/"}

_VERSION = "2.0.0"

"""
Step 3: Change the class name to correspond to your <Your_Dataset_Name> ex: "ChemProtDataset".

Then, fill all relevant information to `BuilderConfig` which populates information about the class. You may have multiple builder configs (ex: a large dataset separated into multiple partitions) if you populate for different dataset names + descriptions. The following is setup for just 1 dataset, but can be adjusted.

NOTE - train/test/dev splits can be handled in `_split_generators`.
"""


class MashQA(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    VERSION = datasets.Version(_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASETNAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME

    """
    Step 4: Populate "information" about the dataset that creates a skeletal
    structure for an example within the dataset looks like.

    The following data structures are useful:

    datasets.Features - An instance that defines all descriptors within a
    feature in an arbitrary nested manner; the "feature" class must strictly
    adhere to this format.

    datasets.Value - the type of the data structure (ex: useful for text,
    PMIDs)

    datasets.Sequence - for information that must be in a continuous sequence
    (ex: spans in the text, offsets)

    An example is as follows for an ENTITY + RELATION dataset
    """

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "answer_start": datasets.Value("int32"),
                            "answer_span": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                        }
                    ),
                    "is_consecutive": datasets.Value("bool"),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    # downloading from google drive solution: https://github.com/huggingface/datasets/issues/418
    def _get_drive_url(self, url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """
        Step 5: Download and extract the dataset.

        For each config name, run `download_and_extract` from the dl_manager;
        this will download and unzip any files within a cache directory,
        specified by `data_dir`.

        `download_and_extract` can accept an iterable object and return the same structure with the url replaced with the path to local files:

        ex: output = dl_manager.download_and_extract(
        {"data1:" "url1", "data2": "url2"})

        output
        >> {"data1": "path1", "data2": "path2"}

        Nested zip files can be cached also, but make sure to save their
        path.

        Populate "SplitGenerator" with `name` and `gen_kwargs`. Note:

        - `name` can be: datasets.Split.<TRAIN/TEST/VALIDATION> or a string
        - all keys in `gen_kwargs` can be passed to `_generate_examples()`. If your dataset has multiple files, you can make a separate key for each file, as shown below:

        """

        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(self._get_drive_url(my_urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "consec_file": os.path.join(data_dir, "train_webmd_squad_v2_consec.json"),
                    "full_file": os.path.join(data_dir, "train_webmd_squad_v2_full.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir,
                    "consec_file": os.path.join(data_dir, "test_webmd_squad_v2_consec.json"),
                    "full_file": os.path.join(data_dir, "test_webmd_squad_v2_full.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir,
                    "consec_file": os.path.join(data_dir, "val_webmd_squad_v2_consec.json"),
                    "full_file": os.path.join(data_dir, "val_webmd_squad_v2_full.json"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, consec_file, full_file, split):
        """
        Step 6: Create a generator that yields (key, example) of the dataset of interest.

        The arguments to this function come from `gen_kwargs` returned in `_split_generators()`

        The goal of this function is to perform operations on any of the keys
        of `gen_kwargs` that allow you to extract and process the data.

        The following skeleton does the following:

        - "extracts" abstracts
        - "extracts" entities, assuming the output is of the form specified in `_info`
        - "extracts" relations, assuming similarly the output in the form specified in `_info`.

        An assumption in this pseudo code is that the abstract, entity, and
        relation file all have linking keys.
        """
        if self.config.name == _DATASETNAME:
            logger.info("generating examples from = %s", filepath)
            files = [full_file, consec_file]
            key = 0
            # first file is not consecutive, second is consecutive
            for file_type_id, file_name in enumerate(files):
                with open(file_name, encoding="utf-8") as f:
                    mashqa = json.load(f)
                    for document in mashqa["data"]:
                        title = document.get("title", "")
                        for paragraph in document["paragraphs"]:
                            context = paragraph["context"]
                            for qa in paragraph["qas"]:
                                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                                answer_span = [answer["answer_spans"] for answer in qa["answers"]]
                                answer_text = [answer["text"] for answer in qa["answers"]]
                                yield key, {
                                    "title": title,
                                    "context": context,
                                    "question": qa["question"],
                                    "id": qa["id"],
                                    "answers": {
                                        "answer_start": answer_starts,
                                        "answer_span": answer_span,
                                        "text": answer_text,
                                    },
                                    "is_consecutive": file_type_id,
                                }
                                key += 1
