import re
from pathlib import Path
import json
from openai import AsyncAzureOpenAI
from mistralai import Mistral
from dotenv import load_dotenv
import os
from typing import Callable, Any, List
import math
import asyncio

from src.generation import prompts as p
from src.datasets import write_examples, write_variable_to_config

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
    client = get_llm_client(api)
    labels, id2label, label2id = create_entity_list(entities, dir_path)
    completion_func = get_completion_func(api)
    asyncio.run(
        create_dataset(
            completion_func=completion_func,
            llm_client=client,
            entities=entities,
            labels=labels,
            nb_samples=nb_samples,
            dir_path=dir_path,
            language=language,
        )
    )
    print("Generation Done")


def get_llm_client(api: str):
    """
    Creates a client instance to make API calls

    Args:
        str: name of the API to use

    Returns:
        Client: client of the API type to make API calls with
    """
    if api == "mistral":
        return get_mistral_instance()
    elif api == "openai":
        return get_async_openai_instance()
    else:
        raise ValueError("API type is not supported")


def get_async_openai_instance() -> AsyncAzureOpenAI:
    """
    Creates an instance of AzureOpenAI client

    Returns:
        AsyncAzureOpenAI: client that enables to make calls to OpenAI API
    """
    client = AsyncAzureOpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    )
    return client


def get_mistral_instance():
    """
    Creates an instance of Mistral client

    Returns:
        Mistral: client that enables to make calls to Mistral API
    """
    client = Mistral(api_key=os.getenv("MISTRAL_KEY"))
    return client


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
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")

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


def get_completion_func(api: str):
    """
    Gets the function that handles the API completion call

    Args:
        api (str): name of the API to use

    Returns:
        Callable[[Client, str, str], str]: function that makes the API call and returns a string
    """
    if api == "mistral":
        return call_mistral_api
    elif api == "openai":
        return call_openai_api
    else:
        raise ValueError("API type is not supported")


async def create_dataset(
    completion_func: Callable[[Any, str, str], str],
    llm_client: Any,
    entities: List[str],
    labels: List[str],
    nb_samples: int,
    dir_path: str,
    language: str,
):
    """
    Iteratively makes API calls until the given number of samples is reached.

    Args:
        completion_func (Callable[[Client, str, str], str]): function that makes the API call and returns a string
        llm_client (Client): client that enables to make calls to the API
        entities (List[str]): list of entities that need to be covered by the dataset
        labels (List[str]): list of labels that can be assigned to the tokens
        nb_samples (int): number of samples to generate
        dir_path (str): path to the dir to save the examples
        language (str): language of the generated dataset
    """

    async def get_completion(i, total_calls, file_path):
        print(f"Making API call: {i+1}/{total_calls}")
        await asyncio.sleep(i + 1.1)

        retry_count = 0
        while True:
            try:
                result = await call_api_async(
                    completion_func, llm_client, entities, labels, language
                )
                sentences = clean_output(result)
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

    async def get_completions():
        total_calls = math.ceil(nb_samples / 50)
        file_path = Path(dir_path) / "dataset.txt"
        tasks = [get_completion(i, total_calls, file_path)
                 for i in range(total_calls)]
        await asyncio.gather(*tasks)

    await get_completions()


async def call_api_async(
    completion_func: Callable[[Any, str, str], str],
    llm_client: [Any],
    entities: list[str],
    labels: list[str],
    language: str,
    examples: list[str] | str = "No examples given",
) -> str:
    """
    Uses the AzureOpenAI client to make a call to the OpenAI API to generate the synthetic data.

    Args:
        completion_func (Callable[[Client, str, str], str]): function that makes the API call and returns a string
        llm_client (Client): client used to make the API call
        entities (list[str]): list of entities that need to be covered by the dataset
        labels (list[str]): list of labels that can be assigned to the tokens
        examples (list[str] | str): (list of) example(s) to further help the LLM
        language (str): language of the generated dataset

    Returns:
        str: the completion returned by the API
    """
    system_prompt = p.SYSTEM_PROMPT.format(
        examples=examples,
        labels=labels,
        formatting_guides=p.TUPLE_FORMAT,
        language=language,
    )
    user_prompt = p.USER_PROMPT.format(entities=entities)
    completion = await completion_func(llm_client, system_prompt, user_prompt)
    return completion


async def call_openai_api(
    client: AsyncAzureOpenAI, system_prompt: str, user_prompt: str
):
    """
    Sends a completion request to OpenAI API user an AzureOpenAI client.

    Args:
        client (AzureOpenAI): client used to make the API call
        system_prompt (str): the system prompt for the API call
        user_prompt (str): the user prompt for the API call

    Returns:
        str: the completion returned by the OpenAI API
    """
    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_NAME"),
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}"},
        ],
        temperature=0.8,
    )

    return completion.choices[0].message.content


async def call_mistral_api(client: Mistral, system_prompt: str, user_prompt: str):
    """
    Sends a completion request to Mistral API user a Mistral client.

    Args:
        client (Mistral): client used to make the API call
        system_prompt (str): the system prompt for the API call
        user_prompt (str): the user prompt for the API call

    Returns:
        str: the completion returned by the Mistral API
    """
    try:
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.complete(
                model=os.getenv("MISTRAL_MODEL_NAME"),
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{user_prompt}"},
                ],
                temperature=0.8,
            ),
        )
    except Exception as e:
        print(e)

    return completion.choices[0].message.content


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
