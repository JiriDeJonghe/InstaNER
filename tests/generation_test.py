from pathlib import Path
import os

from src.generation.generate_dataset import generate_dataset
from src.datasets import read_variable_from_config


def test_generation_with_mistral():
    entities = ["Person", "Location", "Organization"]
    root = Path(__file__).parents[1]
    dir_path = Path.joinpath(root, "experiments", "english", "test_mistral")
    api = "mistral"
    nb_samples = 50

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if os.path.isfile(Path.joinpath(dir_path, "dataset.txt")):
        os.remove(Path.joinpath(dir_path, "dataset.txt"))

    generate_dataset(
        entities=entities,
        dir_path=dir_path,
        api=api,
        nb_samples=nb_samples,
        language="english",
    )

    labels = read_variable_from_config(dir_path, "labels")
    label2id = read_variable_from_config(dir_path, "label2id")
    id2label = read_variable_from_config(dir_path, "id2label")

    assert os.path.isdir(dir_path)
    assert os.path.isfile(Path.joinpath(dir_path, "dataset.txt"))

    assert isinstance(labels, list)
    assert isinstance(id2label, dict)
    assert isinstance(label2id, dict)

    assert len(labels) == len(id2label.keys())
    assert len(labels) == len(label2id.keys())


def test_generation_with_openai():
    entities = ["Person", "Location", "Organization"]
    root = Path(__file__).parents[1]
    dir_path = Path.joinpath(root, "experiments", "english", "test_openai")
    api = "openai"
    nb_samples = 50

    generate_dataset(
        entities=entities,
        dir_path=dir_path,
        api=api,
        nb_samples=nb_samples,
        language="english",
    )

    labels = read_variable_from_config(dir_path, "labels")
    label2id = read_variable_from_config(dir_path, "label2id")
    id2label = read_variable_from_config(dir_path, "id2label")

    assert os.path.isdir(dir_path)
    assert os.path.isfile(Path.joinpath(dir_path, "dataset.txt"))

    assert isinstance(labels, list)
    assert isinstance(id2label, dict)
    assert isinstance(label2id, dict)

    assert len(labels) == len(id2label.keys())
    assert len(labels) == len(label2id.keys())
