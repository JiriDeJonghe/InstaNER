from openai import OpenAI, AsyncOpenAI
from mistralai import Mistral
from google import genai
from google.genai import Client, types
from typing import Optional
import os
import asyncio

def get_completion(api: str, messages: list[dict[str, str]], tools: Optional[list[dict[str, str]]] = None) -> str:
    """
    Gets an LLM completion

    Args:
        api (str): the name of the API to use (can be OpenAI, Google, or Mistral)
        messages (list[dict[str, str]]): the messages to send to the LLM
        tools (list[dict[str, str]], Optional): the tools the LLM can use

    Returns:
        str: completion of the LLM
    """
    client = get_llm_client(api, False)
    func = get_completion_func(api)
    completion = func(client, messages, tools)

    return completion


async def async_get_completion(api: str, messages: list[dict[str, str]], tools: Optional[list[dict[str, str]]] = None) -> str:
    """
    Gets an LLM completion asynchronously

    Args:
        api (str): the name of the API to use (can be OpenAI, Google, or Mistral)
        messages (list[dict[str, str]]): the messages to send to the LLM
        tools (list[dict[str, str]], Optional): the tools the LLM can use

    Returns:
        str: completion of the LLM
    """
    client = get_llm_client(api, True)
    func = get_async_completion_func(api)
    completion = await func(client, messages, tools)

    return completion


def get_llm_client(api: str, async_: bool = False):
    """
    Creates a client instance to make API calls

    Args:
        api (str): the name of the API to use (can be OpenAI, Google, or Mistral)
        async_ (boolean): true if wanting an async client

    Returns:
        Client: client of the API type to make API calls with
    """
    if api == "mistral":
        return get_mistral_instance()
    elif api == "openai":
        if async_:
            return get_async_openai_instance()
        return get_openai_instance()
    elif api == "google":
        return get_google_instance()
    else:
        raise ValueError("API type is not supported")


def get_openai_instance() -> OpenAI:
    """
    Creates an instance of OpenAI client

    Returns:
        OpenAI: client that enables to make calls to OpenAI API
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    return client


def get_async_openai_instance() -> AsyncOpenAI:
    """
    Creates an instance of an async OpenAI client

    Returns:
        AsyncOpenAI: async client that enables to make calls to OpenAI API
    """
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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

def get_google_instance() -> Client:
    """
    Creates an instance of Google Gemini client

    Returns:
        Client: client that enables to make calls to Gemini API
    """
    client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    return client


def get_completion_func(api: str):
    """
    Gets the function that handles the API completion call

    Args:
        api (str): the name of the API to use (can be OpenAI, Google, or Mistral)

    Returns:
        Callable[[Client, str, str], str]: function that makes the API call and returns a string
    """
    if api == "mistral":
        return call_mistral_api
    elif api == "openai":
        return call_openai_api
    elif api == "google":
        return call_google_api
    else:
        raise ValueError("API type is not supported")


def call_mistral_api(client: Mistral, messages: list[dict], tools: list[dict]) -> Optional[str]:
    """
    Sends a completion request to Mistral API using a Mistral client.

    Args:
        client (Mistral): client used to make the API call
        messages (list[dict]): the messages to send to the LLM
        tools (list[dict]): the tools that can be used by the LLM

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
    client: OpenAI, messages: list[dict], tools: list[dict]
) -> Optional[str]:
    """
    Sends a completion request to OpenAI API using an OpenAI client.

    Args:
        client (OpenAI): client used to make the API call
        messages (list[dict]): the messages to send to the LLM
        tools (list[dict]): the tools that can be used by the LLM

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
        print(f"Error: {e}")
        return

    return completion.choices[0].message


def call_google_api(
    client: Client, messages: list[dict[str, str]], tools: list[dict]
) -> Optional[str]:
    """
    Sends a completion request to Gemini API using an Client.

    Args:
        client (Client): client used to make the API call
        messages (list[dict]): the messages to send to the LLM
        tools (list[dict]): the tools that can be used by the LLM

    Returns:
        str: the completion returned by the OpenAI API
    """
    contents = ""
    for message in messages:
        contents += f"{message["role"]}: {message["content"]}\n"

    tools = types.GenerateContentConfig(tools=convert_openai_tools_to_gemini(tools))

    try:
        completion = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL_NAME"),
            contents=contents,
            config=tools
        )
    except Exception as e:
        print(e)

    return completion.candidates[0].content.parts[0]


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
    elif api == "google":
        return async_call_google_api
    else:
        raise ValueError("API type is not supported")


async def async_call_mistral_api(client: Mistral, messages: list[dict], tools: list[dict] = None):
    """
    Asynchronously sends a completion request to Mistral API using a Mistral client.

    Args:
        client (Mistral): client used to make the API call
        messages (list[dict]): the messages to pass on to the LLM
        tools (list[dict], Optional): the tools the LLM can use

    Returns:
        str: the completion returned by the Mistral API
    """
    try:
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.complete(
                model=os.getenv("MISTRAL_MODEL_NAME"),
                messages=messages,
                temperature=0.5,
                tools=tools,
            ),
        )
    except Exception as e:
        print(e)

    return completion.choices[0].message


async def async_call_openai_api(
    client: AsyncOpenAI, messages: list[dict], tools: list[dict] = None
):
    """
    Asynchronoulsy sends a completion request to OpenAI API using an AsyncOpenAI client.

    Args:
        client (AsyncOpenAI): client used to make the API call
        messages (list[dict]): the messages to pass on to the LLM
        tools (list[dict], Optional): the tools the LLM can use

    Returns:]
        str: the completion returned by the OpenAI API
    """
    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_NAME"),
        messages=messages,
        temperature=0.5,
        tools=tools,
    )

    return completion.choices[0].message


async def async_call_google_api(
    client: Client, messages: list[dict[str, str]], tools: Optional[list[dict]] = []
) -> Optional[str]:
    """
    Asynchronoulsy sends a completion request to Gemini API using an Client.

    Args:
        client (Client): client used to make the API call
        messages (list[dict]): the messages to send to the LLM
        tools (list[dict]): the tools that can be used by the LLM

    Returns:
        str: the completion returned by the OpenAI API
    """
    contents = ""
    for message in messages:
        contents += f"{message["role"]}: {message["content"]}\n"

    if tools:
        config = types.GenerateContentConfig(temperature = 0.5, tools=convert_openai_tools_to_gemini(tools))
    else:
        config = types.GenerateContentConfig(temperature = 0.5)

    try:
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=os.getenv("GEMINI_MODEL_NAME"),
                contents=contents,
                config=config
            ),
        )
    except Exception as e:
        print(e)

    return completion.candidates[0].content.parts[0]

def convert_openai_tools_to_gemini(openai_tools: str) -> list[types.Tool]:
    """
    Converts the OpenAI formatted tools to Gemini format

    Args:
        openai_tools (str): the openai formatted tool list

    Returns:
        list[Tool]: the tool declarations in Gemini format
    """
    gemini_tools: list[types.Tool] = []

    for tool in openai_tools:
        function = tool.get("function")
        if not function:
            continue
        
        function_declaration = types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=convert_openai_schema_to_gemini(function.get("parameters", {}))
            )

        gemini_tools.append(types.Tool(function_declarations=[function_declaration]))

    return gemini_tools


def convert_openai_schema_to_gemini(openai_schema: str) -> types.Schema:
    """
    Converts the OpenAI schema for tools to Gemini format 

    Args:
        openai_schema (str): the original openai schema

    Returns:
        list[FunctionDeclaration]: the function declarations formatted to Gemini
    """
    schema_type = openai_schema.get("type", "object")

    if schema_type == "array":
        items_schema = openai_schema.get("items", {})
        return types.Schema(
            type="array",
            items=convert_openai_schema_to_gemini(items_schema)
        )

    return types.Schema(
        type=schema_type,
        properties={
            key: convert_openai_schema_to_gemini(value) if isinstance(value, dict) else types.Schema(type=value) for key, value in openai_schema.get("properties", {}).items()
        },
        required=openai_schema.get("required", [])
    )

