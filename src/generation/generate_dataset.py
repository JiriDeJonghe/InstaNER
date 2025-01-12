import re
from pathlib import Path
import json
from dotenv import load_dotenv
import os
from typing import Callable, Any, List
import math
import asyncio

from src.generation import prompts as p
from src.datasets import write_examples, write_variable_to_config
from src.llm.api import async_get_completion

load_dotenv()


def generate_dataset(
    entities: list[str], dir_path: str, api: str, nb_samples: int, language: str
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Generates synthetic data for the training of NER

    Args:
        entities (list[str]): different types of entities that should be covered by the dataset
        dir_path (str): path to the file to save the generated samples
        api (str): string indicating the kind of API to use to generate the dataset
        nb_samples (int): number of samples that need to be generated for the dataset
        language (str): language of the generated dataset

    Returns:
        list[str]: list of labels that can be assigned to the tokens
        dict[str, str]: mapping of IDs to labels
        dict[str, str]: mapping of labels to IDs
    """
    print("\n")
    print("Starting Generation")
    print("-" * 32)

    root = Path(__file__).parents[2]
    dir_path = Path.joinpath(root, "experiments", language, dir_path)
    print(dir_path)
    if dir_path.exists():
        print(
            "Model directory already exists. Aborting."
        )
        return
    elif not dir_path.exists():
        dir_path.mkdir(parents=True)

    labels, id2label, label2id = create_entity_list(entities, dir_path)
    system_prompt = p.SYSTEM_PROMPT.format(
        examples="",
        labels=labels,
        formatting_guides=p.TUPLE_FORMAT,
        language=language,
    )
    user_prompt = p.USER_PROMPT.format(entities=entities)

    asyncio.run(
        create_dataset(
            api=api,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            nb_samples=nb_samples,
            dir_path=dir_path,
            language=language,
        )
    )
    print("Generation Done")


def create_entity_list(entities: list[str], path: str) -> list[str]:
    """
    Creates a list of entity values given the entities that are given:
    B- indicated the beginning of an entity
    I- indicated the token is inside the same entity
    0- indicates the token doesn't correspond to any entity

    Args:
        entities (list[str]): list of entities that need to be covered by the dataset
        path (str): path to the dir to save the labels and the mappings

    Returns:
        list[str]: list of labels that can be assigned to the tokens
        dict[str, str]: mapping of IDs to labels
        dict[str, str]: mapping of labels to IDs
    """
    labels = ["0"]
    for entity in entities:
        labels.append(f"B-{entity.upper()}")
        labels.append(f"I-{entity.upper()}")

    id2label = {}
    label2id = {}
    for id, entity in enumerate(labels):
        id2label[id] = entity
        label2id[entity] = id

    write_variable_to_config(path, {"entities": entities})
    write_variable_to_config(path, {"labels": labels})
    write_variable_to_config(path, {"id2label": id2label})
    write_variable_to_config(path, {"label2id": label2id})

    return labels, id2label, label2id


async def create_dataset(
    api: str,
    system_prompt: str,
    user_prompt: str,
    nb_samples: int,
    dir_path: str,
    language: str,
):
    """
    Iteratively makes API calls until the given number of samples is reached.

    Args:
        api (str): identifier of the LLM to use
        system_prompt (str): the system prompt to use for the completion
        user_prompt (str): the user prompt to use for the completion
        nb_samples (int): number of samples to generate
        dir_path (str): path to the dir to save the examples
        language (str): language of the generated dataset
    """

    async def make_call(i, total_calls, file_path):
        print(f"Making API call: {i+1}/{total_calls}")
        await asyncio.sleep(i + 1.1)

        retry_count = 0
        while True:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                result = await async_get_completion(
                    api, messages
                )
                print("Response Received")
                sentences = clean_output(result.content)
                await asyncio.get_event_loop().run_in_executor(
                    None, write_examples, file_path, sentences
                )
                break
            except Exception as e:
                retry_count += 1
                wait_time = 2**retry_count
                print(f"Failed generating samples: {e}")
                print(f"Sleeping {wait_time} seconds")
                await asyncio.sleep(wait_time)
                print("Trying again...")

    async def make_calls():
        total_calls = math.ceil(nb_samples / 25)
        file_path = Path(dir_path) / "dataset.txt"
        tasks = [make_call(i, total_calls, file_path)
                 for i in range(total_calls)]
        await asyncio.gather(*tasks)

    await make_calls()


def clean_output(string: str) -> list[dict[str, str]]:
    """
    Cleans the output generated by the LLM in case it generated more than just the JSON object

    Args:
        str: the completion returned by the OpenAI API

    Returns:
        list[dict[str, str]]: list of dictionaries containing the examples
    """
    python_string = r"```json(.*?)```"
    result_code = re.search(python_string, string, re.IGNORECASE | re.DOTALL)
    result_json = json.loads(result_code.group(1))
    try:
        sentences = result_json["sentences"]
        return sentences
    except Exception:
        ValueError(f"Failed parsing the LLM result: {result_code}")
