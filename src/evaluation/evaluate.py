from transformers import pipeline
from pathlib import Path
from src.evaluation.eval_metrics import (
    compute_metrics,
)
from src.datasets import (
    read_variable_from_config,
    write_variable_to_config,
    load_train_test_dataset,
)


def evaluate_model(dir_path: Path, language: str, model_type: str = "transformer"):
    """
    Evaluates the model on the given dataset. Can be either supplied as a NERDataset object or a string with the name of the dataset

    Args:
        dir_path (Path): path to the dir containing the model and dataset to evaluate
        language (str): language of the generated dataset and model
        model_type (str): type of the model to evaluate

    Returns:
        dict[str]: contains the computed metrics
    """
    if model_type == "transformer":
        return evaluate_transformer_model(dir_path, language)


def evaluate_transformer_model(dir_path: Path, language: str):
    """
    Evaluates the model on the given datase. Can be either supplied as a NERDataset object or a string with the name of the dataset

    Args:
        dir_path (Path): path to the dir containing the model and dataset to evaluate
        language (str): language of the generated dataset and model

    Returns:
        dict[str]: contains the computed metrics
    """
    print("Starting Evaluation")
    print("-" * 32)

    root = Path(__file__).parents[2]
    dir_path = Path.joinpath(root, "experiments", language, dir_path)

    if not Path.exists(Path.joinpath(dir_path)):
        raise Exception("Directory does not exist")

    model_path = Path.joinpath(dir_path, "model")
    classifier = pipeline("ner", model=model_path)

    try:
        label2id = read_variable_from_config(dir_path, "label2id")
        id2label = read_variable_from_config(dir_path, "id2label")
        _, test_dataset = load_train_test_dataset(dir_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    converted_test_sentences = test_dataset.get_converted_sentences()
    predictions = []
    for sentence in converted_test_sentences["tokens"]:
        predictions.append(classifier(sentence))
    truths = converted_test_sentences["ner_tags"]

    evaluation = compute_metrics((predictions, truths), label2id, id2label)
    overall_eval = evaluation["overall"]
    print("Precision: ", overall_eval["precision"])
    print("Recall: ", overall_eval["recall"])
    print("F1: ", overall_eval["f1"])
    print("Accuracy: ", overall_eval["accuracy"])

    write_variable_to_config(dir_path, {"evluation": evaluation})
    return evaluation
