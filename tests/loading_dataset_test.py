from pathlib import Path
from transformers import AutoTokenizer, BatchEncoding
import os
import pytest

from src.datasets import (
    load_train_test_dataset,
    write_variable_to_config,
    NERDataset,
)
from src.generation.generate_dataset import generate_dataset

ROOT = Path(__file__).parents[1]
DIR_PATH = Path.joinpath(ROOT, "experiments", "english", "test")
FILE_PATH = Path.joinpath(DIR_PATH, "dataset.txt")
tokenizer_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


@pytest.fixture(scope="function", autouse=True)
def create_examples():
    if not os.path.isdir(DIR_PATH):
        os.makedirs(DIR_PATH)
    if not os.path.isfile(FILE_PATH):
        generate_dataset(
            entities=["Person", "Location", "Organization"],
            dir_path=DIR_PATH,
            api="mistral",
            nb_samples=50,
            language="english",
        )
    write_variable_to_config(
        dir_path=DIR_PATH, config_update_dict={"tokenizer": tokenizer_name}
    )


def test_load_dataset_transformer():
    train_dataset, test_dataset = load_train_test_dataset(dir_path=DIR_PATH)

    assert isinstance(train_dataset, NERDataset)
    assert isinstance(test_dataset, NERDataset)

    assert isinstance(train_dataset.data, list)
    assert isinstance(test_dataset.data, list)

    assert isinstance(train_dataset.label2id, dict)
    assert isinstance(test_dataset.label2id, dict)

    assert isinstance(train_dataset.converted_sentences, list)
    assert isinstance(test_dataset.converted_sentences, list)

    example = train_dataset.__getitem__(0)
    assert isinstance(example, BatchEncoding)
    assert isinstance(example.get("input_ids"), list)
    assert isinstance(example.get("attention_mask"), list)
    assert isinstance(example.get("labels"), list)
    assert len(example.get("labels")) == len(example.get("input_ids"))
    assert len(example.get("labels")) == len(example.get("attention_mask"))
