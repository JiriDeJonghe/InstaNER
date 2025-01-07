from pathlib import Path
import os
from transformers import pipeline

from src.datasets import read_variable_from_config
from src.training.train_transformer import train_model

ENTITIES = ["Person", "Location", "Organization"]
ROOT = Path(__file__).parents[1]
DIR_PATH = Path.joinpath(ROOT, "experiments", "english", "test")
FILE_PATH = Path.joinpath(DIR_PATH, "test_dataset.txt")
MODEL_PATH = Path.joinpath(DIR_PATH, "model")
labels = read_variable_from_config(DIR_PATH, "labels")
label2id = read_variable_from_config(DIR_PATH, "label2id")
id2label = read_variable_from_config(DIR_PATH, "id2label")
test_sentence = (
    "This is a test sentence. It was written by Jiri De Jonghe. Jiri lives in Belgium."
)


def test_train_model():
    train_model(
        dir_path=DIR_PATH,
        language="english",
    )

    assert os.path.isdir(MODEL_PATH)
    required_files = ["config.json", "tokenizer_config.json"]
    for file in required_files:
        assert os.path.isfile(Path.joinpath(MODEL_PATH, file))

    classifier = pipeline("ner", model=MODEL_PATH)
    result = classifier(test_sentence)
    assert isinstance(result, list)
