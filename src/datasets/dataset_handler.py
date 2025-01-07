import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pathlib import Path
import random
from typing import Dict, Any
import os

from .ner_dataset import NERDataset


def load_train_test_dataset(dir_path: str, type: str = "transformer"):
    """
    Loads the train and test dataset for an experiment located at dir_path.
    Converts the dataset in a suitable format for the model training.

    Args:
        dir_path (str): path to the dir for the experiment
        type (str): type of the model to generate the datasets for

    Returns:
        NERDataset: training dataset
        NERDataset: test dataset
    """
    # Reading in examples that were generated in sentences.txt
    file_path = Path.joinpath(dir_path, "dataset.txt")
    sentences = read_examples(file_path)

    # Preprocess the tokens field, reproducable results using seed
    seed = read_variable_from_config(dir_path, "seed")
    if seed is None:
        seed = random.randint(0, 100000) if seed is None else seed
        write_variable_to_config(dir_path, {"seed": seed})

    train_sentences, test_sentences = train_test_split(
        sentences, test_size=0.2, random_state=seed
    )

    if type == "transformer":
        train_dataset, test_dataset = load_train_test_dataset_transformer(
            dir_path, train_sentences, test_sentences
        )

    return train_dataset, test_dataset


def load_train_test_dataset_transformer(
    dir_path: str, train_sentences: list[str], test_sentences: list[str]
) -> tuple[NERDataset, NERDataset]:
    """
    Constructs the training and test dataset for a transformer model

    Args:
        dir_path (str): path to the dir for the experiment
        train_sentences (list[str]): list of training examples
        test_sentences (list[str]): list of test examples

    Returns:
        NERDataset: training dataset
        NERDataset: test dataset
    """
    # Load tokenizer
    tokenizer_name = read_variable_from_config(dir_path, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load labels
    label2id = read_variable_from_config(dir_path, "label2id")

    # Create NERDataset
    train_dataset = NERDataset(train_sentences, tokenizer, label2id)
    test_dataset = NERDataset(test_sentences, tokenizer, label2id)

    return train_dataset, test_dataset


def read_examples(file_path: str) -> list[str]:
    """
    Reads examples from the generated dataset file and loads them into a list

    Args:
        file_path (str): path to the file to read the sentences from

    Returns:
        list[str]: sentences to train the model on

    Raises:
        Exception: if the writing to the path fails
    """
    try:
        with open(file_path, "r") as f:
            sentences = f.readlines()
            sentences = [eval(sentence) for sentence in sentences]
        return sentences

    except Exception:
        raise Exception(f"Error reading file: {file_path}")


def write_examples(file_path: str, sentences: str) -> None:
    """
    Writes the generated examples to the dataset file

    Args:
        file_path (str): path to the file to write the sentences away to
        list[dict[str, str]]: list of dictionaries containing the examples
    """
    try:
        with open(file_path, "a+") as file:
            for sentence in sentences:
                sentence_str = str(sentence)
                file.write(sentence_str)
                file.write("\n")

    except Exception:
        raise Exception(f"Error writing file: {file_path}")


def write_variable_to_config(dir_path: str, config_update_dict: Dict[str, Any]) -> None:
    """
    Updates the config file by providing new variables and their value

    Args:
        dir_path (str): path to the dir where the seed.txt file is
        config_update_dict (Dict[str, Any]): dictionary of variables to add to the config file
    """
    config_file_path = Path(dir_path) / "config.json"
    if os.path.isfile(config_file_path):
        with open(config_file_path, "r") as f:
            config = json.load(f)
            config.update(config_update_dict)
    else:
        config = config_update_dict

    with open(config_file_path, "w+") as f:
        json.dump(config, f)


def read_variable_from_config(dir_path: str, key_name: str) -> str:
    """
    Retrieves a variable from the config file

    Args:
        dir_path (str): path to the dir where the config file is
        key_name (str): name of the variable to retrieve

    Returns:
        str | None: value of the variable. None if not found
    """
    config_file_path = Path(dir_path) / "config.json"
    if os.path.isfile(config_file_path):
        with open(config_file_path, "r") as f:
            config = json.load(f)
            return config.get(key_name, None)
