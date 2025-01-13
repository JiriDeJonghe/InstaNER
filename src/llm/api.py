from openai import OpenAI, AsyncOpenAI
from mistralai import Mistral
from typing import List, Dict
import os
import asyncio

def get_completion(api: str, messages: List[Dict], tools: List[Dict] = None) -> str:
    """
    Gets an LLM completion

    Args:
        api (str): the name of the API to use (can be OpenAI or Mistral)
        messages (List[Dict]): the messages to send to the LLM
        tools (List[Dict], Optional): the tools the LLM can use

    Returns:
        str: completion of the LLM
    """
    client = get_llm_client(api, False)
    func = get_completion_func(api)
    completion = func(client, messages, tools)

    return completion


async def async_get_completion(api: str, messages: List[str], tools: List[Dict] = None):
    """
    Gets an LLM completion asynchronously

    Args:
        api (str): the name of the API to use (can be OpenAI or Mistral)
        messages (List[Dict]): the messages to send to the LLM
        tools (List[Dict], Optional): the tools the LLM can use

    Returns:
        str: completion of the LLM
    """
    client = get_llm_client(api, True)
    func = get_async_completion_func(api)
    completion = await func(client, messages, tools)

    return completion


def get_llm_client(api: str, async_: str = False):
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
        if async_:
            return get_async_openai_instance()
        return get_openai_instance()
    else:
        raise ValueError("API type is not supported")


def get_openai_instance() -> OpenAI:
    """
    Creates an instance of OpenAI client

    Returns:
        OpenAI: client that enables to make calls to OpenAI API
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    return client


def get_async_openai_instance() -> AsyncOpenAI:
    """
    Creates an instance of AzureOpenAI client

    Returns:
        AsyncAzureOpenAI: async client that enables to make calls to OpenAI API
    """
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    return client


def get_mistral_instance() -> Mistral:
    """
    Creates an instance of Mistral client

    Returns:
        Mistral: client that enables to make calls to Mistral API
    """
    client = Mistral(api_key=os.getenv("MISTRAL_KEY"))
    return client


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


def call_mistral_api(client: Mistral, messages: List[Dict], tools: List[Dict]):
    """
    Sends a completion request to Mistral API user a Mistral client.

    Args:
        client (Mistral): client used to make the API call
        messages (List[Dict]): the messages to send to the LLM
        tools (List[Dict]): the tools that can be used by the LLM

    Returns:
        str: the completion returned by the Mistral API
    """
    try:
        completion = client.chat.complete(
            model=os.getenv("MISTRAL_MODEL_NAME"),
            messages=messages,
            tools=tools,
        )
    except Exception as e:
        print(e)

    return completion.choices[0].message


def call_openai_api(
    client: OpenAI, messages: List[Dict], tools: List[Dict]
):
    """
    Sends a completion request to OpenAI API user an AzureOpenAI client.

    Args:
        client (OpenAI): client used to make the API call
        messages (List[Dict]): the messages to send to the LLM
        tools (List[Dict]): the tools that can be used by the LLM

    Returns:
        str: the completion returned by the OpenAI API
    """
    try:
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME"),
            messages=messages,
            tools=tools,
        )
    except Exception as e:
        print(e)

    return completion.choices[0].message


def get_async_completion_func(api: str):
    """
    Gets the function that handles the API completion call

    Args:
        api (str): name of the API to use

    Returns:
        Callable[[Client, str, str], str]: function that makes the API call and returns a string
    """
    if api == "mistral":
        return async_call_mistral_api
    elif api == "openai":
        return async_call_openai_api
    else:
        raise ValueError("API type is not supported")


async def async_call_mistral_api(client: Mistral, messages: List[Dict], tools: List[Dict] = None):
    """
    Sends a completion request to Mistral API user a Mistral client.

    Args:
        client (Mistral): client used to make the API call
        messages (List[Dict]): the messages to pass on to the LLM

    Returns:
        str: the completion returned by the Mistral API
    """
    try:
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.complete(
                model=os.getenv("MISTRAL_MODEL_NAME"),
                messages=messages,
                temperature=0.8,
            ),
        )
    except Exception as e:
        print(e)

    return completion.choices[0].message


async def async_call_openai_api(
    client: AsyncOpenAI, messages: List[Dict], tools: List[Dict] = None
):
    """
    Sends a completion request to OpenAI API user an AzureOpenAI client.

    Args:
        client (AzureOpenAI): client used to make the API call
        messages (List[Dict]): the messages to pass on to the LLM
        tools (List[Dict], Optional): the tools the LLM can use

    Returns:
        str: the completion returned by the OpenAI API
    """
    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_NAME"),
        messages=messages,
        temperature=0.8,
        tools=tools,
    )

    return completion.choices[0].message
