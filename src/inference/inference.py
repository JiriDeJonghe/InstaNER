from transformers import pipeline
import argparse
from pathlib import Path


def inference(dir_path: str, language: str) -> None:
    """
    Detects the token entities from a sentence using Named Entity Recognition and prints them out to standard output.

    Args:
        dir_path (str): path to the dir containing the model
        language (str): language of the generated dataset and model
    """
    print("Starting Inference")
    print("-" * 32)

    root = Path(__file__).parents[2]
    dir_path = Path.joinpath(root, "experiments", language, dir_path)

    model_path = Path.joinpath(dir_path, "model")
    classifier = pipeline("ner", model=model_path)

    while True:
        print("Enter sentence to perform NER on or type 'q' to quit")
        text = input()

        if text == "q":
            break

        result = classifier(text)
        if all(entity.get("entity") == "0" for entity in result):
            print("No entities found")
            continue
        for entity in result:
            if entity.get("entity") != "0":
                print("-" * 32)
                print(f"Entity: {entity.get('word')}")
                print(f"Type: {entity.get('entity')}")
                print(f"Score: {entity.get('score')}")


def main():
    parser = argparse.ArgumentParser(
        description="Use the trained NER model for inference"
    )

    parser.add_argument(
        "--directory",
        "-d",
        required=True,
        help="Name of the directory associated with this experiment of which to use the model for inference",
    )

    args = parser.parse_args()

    root = Path(__file__).parents[2]
    model_path = Path.joinpath(root, "models", args.model)
    inference(args.text, model_path)


if __name__ == "__main__":
    exit(main())
