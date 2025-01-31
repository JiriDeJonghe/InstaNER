from pathlib import Path
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src.datasets import (
    load_train_test_dataset,
    read_variable_from_config,
    write_variable_to_config,
)


def train_model(
    dir_path: Path,
    language: str = "english",
) -> None:
    """
    Trains a NER model based on the distilbert-base-uncased given the labels.

    Args:
        dir_path (Path): path to the file containing the examples
        language (str): language of the generated dataset and model
   """
    print("\n")
    print("Starting Training")
    print("-" * 32)

    root = Path(__file__).parents[2]
    dir_path = Path.joinpath(root, "experiments", language, dir_path)

    # Load dataset and update config for reproducability
    model_name = get_model_name_from_language(language)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    write_variable_to_config(str(dir_path), {"model": model_name})
    write_variable_to_config(str(dir_path), {"tokenizer": model_name})

    label2id = read_variable_from_config(dir_path, "label2id")
    id2label = read_variable_from_config(dir_path, "id2label")

    train_dataset, test_dataset = load_train_test_dataset(dir_path)

    print("Loaded Dataset")
    print("Training set size: ", len(train_dataset))
    print("Testing set size: ", len(test_dataset))

    # Create batch of examples
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load base model for training
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id.keys()),
        id2label=id2label,
        label2id=label2id,
    )

    print("Loaded Model")

    # Define training arguments
    output_dir = Path(dir_path) / "model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    training_info_dict = {
        "training_args": {
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "weight_decay": training_args.weight_decay,
            "eval_strategy": training_args.eval_strategy,
            "save_strategy": training_args.save_strategy,
            "load_best_model_at_end": training_args.load_best_model_at_end,
        },
    }
    write_variable_to_config(dir_path, training_info_dict)
    print("Arguments Defined")


    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Trainer defined")

    # Train the model
    trainer.train()
    trainer.save_model(output_dir)

    print("-" * 32)
    print("Model Training Complete")


def get_model_name_from_language(language: str):
    """
    Returns the name of the model based on the language

    Args:
        language (str): language of the generated dataset and model

    Returns:
        str: name of the model
    """

    match language:
        case "english":
            return "distilbert-base-uncased"
        case "dutch":
            return "GroNLP/bert-base-dutch-cased"
        case _:
            return "bert-base-multilingual-cased"
