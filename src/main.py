from pathlib import Path
import argparse
import datetime

from src.generation.generate_dataset import generate_dataset
from src.training.train_transformer import train_model
from src.evaluation.evaluate import evaluate_model
from src.inference.inference import inference
from src.llm.agent import run_conversation


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset and automatically train a NER model with the created data"
    )

    parser.add_argument(
        "--agent",
        "-a",
        required=False,
        help="List of entities that should be recognized by the model. If training the model only, make sure the order is the same as when creating the dataset.",
    )

    parser.add_argument(
        "--entities",
        "-e",
        nargs="+",
        required=False,
        help="List of entities that should be recognized by the model. If training the model only, make sure the order is the same as when creating the dataset.",
    )

    parser.add_argument(
        "--language",
        "-l",
        required=False,
        default="english",
        help="Language of the dataset and the model. Only required if you want to create new data",
    )

    parser.add_argument(
        "--api",
        "-ap",
        required=False,
        default="mistral",
        help="Name of the API that is used to generate the dataset. Can be either 'mistral' or 'openai'. Only required if you want to create new data",
    )

    parser.add_argument(
        "--nb_samples",
        "-nb",
        required=False,
        default=1000,
        help="Number of samples that need to be generated for the synthetic dataset. Only required if you want to create new data",
    )

    parser.add_argument(
        "--path",
        "-p",
        required=False,
        default="",
        help="Path to the directory where all the information for this run should be stored",
    )

    parser.add_argument(
        "--steps",
        "-s",
        nargs="+",
        required=False,
        default=["all"],
        help="The steps of the pipeline that you want to run. Can be 'all', 'generation', 'training', 'evaluation', 'inference'. Defaults to 'all'.",
    )

    args = parser.parse_args()
    print(args)

    if args.agent:
        run_conversation()
        return

    try:
        root = Path(__file__).parents[1]
        path = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if not args.path
            else args.path
        )

        if any(x in ["all", "generation"] for x in args.steps):
            generate_dataset(
                entities=args.entities,
                dir_path=path,
                api=args.api,
                nb_samples=int(args.nb_samples),
                language=args.language,
            )
        if any(x in ["all", "training"] for x in args.steps):
            train_model(
                dir_path=path,
                language=args.language,
            )
        if any(x in ["all", "evaluation"] for x in args.steps):
            evaluate_model(path, args.language)
        if any(x in ["all", "inference"] for x in args.steps):
            inference(path, args.language)
    except Exception as e:
        print(f"Error executing pipeline: {str(e)}")


if __name__ == "__main__":
    exit(main())
