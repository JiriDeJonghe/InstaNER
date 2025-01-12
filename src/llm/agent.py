import json
from typing import List, Dict, Any
from openai import OpenAI

from src.llm.api import get_completion
from src.generation.generate_dataset import generate_dataset
from src.training.train_transformer import train_model
from src.evaluation.evaluate import evaluate_model
from src.inference.inference import inference


tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_dataset",
            "description": "Generates synthetic data for the generation of a dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "Different types of entities that should be covered by the dataset",
                        "items": {
                            "type": "string"
                        }
                    },
                    "dir_path": {
                        "type": "string",
                        "description": "Directory to store the data in"
                    },
                    "api": {
                        "type": "string",
                        "description": "Indicator for the kind of API to use to generate the dataset. Can be 'mistral' or 'openai'."
                    },
                    "nb_samples": {
                        "type": "integer",
                        "description": "Number of samples that need to be generated for the dataset"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the generated dataset"
                    },
                },
                "required": ["entities", "dir_path", "api", "nb_samples", "language"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "Trains the model using synthetic data that has been generated",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory to store the data in"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the generated dataset"
                    },
                },
                "required": ["dir_path", "language"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Evaluates the model after it has been trained",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory to store the data in"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the generated dataset"
                    },
                },
                "required": ["dir_path", "language"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inference",
            "description": "Loads the model for inference. Will start an infinite loop to use the model until it is cancelled",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Directory to store the data in"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the generated dataset"
                    },
                },
                "required": ["dir_path", "language"]
            }
        }
    },
]

supported_model_types = ["Named Entity Recognition"]

agent_prompt = f"""
You're a helpful assistent whose goal is to help humans train ML models.

Currently you can help with the following types of models: {supported_model_types}

The following steps need to be done in order to generate a ML model:
1. Create synthetic data on which the model should be trained
2. Training of the model
3. Evaluation of the model - and confirm that it achieves the required results
4. Load it for inference - the human immediately would like to test out the model for inference

After the model has been loaded for inference, your job is done and you say goodbye to the user.

You are given a set of tools that you can use to achieve this task. Only start calling tools when you have all the requirements that you need.
"""


tool_registry = {
    "generate_dataset": generate_dataset,
    "train_model": train_model,
    "evaluate_model": evaluate_model,
    "inference": inference
}

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Executes the tool with the given arguments

    Args:
        tool_name (str): name of the tool to execute
        arguments (Dict[str, Any]): arguments to call the tool with

    Returns:
        str: result of the tool call
    """
    try:
        if tool_name not in tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        result = tool_registry[tool_name] (**arguments)

        if not isinstance(result, str):
            result = json.dumps(result)

        return result
    except Exception as e:
        print(f"Error executing {tool_name}: {str(e)}")
        return f"Error executing {tool_name}: {str(e)}"


def run_conversation():
    """
    Runs an infinite loop for the conversation with the agent
    """
    messages = [
        {"role": "system", "content": agent_prompt},
        {"role": "assistant", "content": "Hello! I'm here to help you train ML models. What type of model would you like to work on today?"}
    ]
    
    while True:
        print("\nAssistant:", messages[-1]["content"])
        
        user_input = input("\nYou (type 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = get_completion(
            api="openai",
            messages=messages,
            tools=tools,
        )

        if response.tool_calls:
            assistant_message = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in response.tool_calls
                ]
            }
            messages.append(assistant_message)

            for tool_call in response.tool_calls:
                print("\nAssistant:", response.content)
                print(f"\nI would like to call the following tool:")
                print(f"Tool: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
                
                confirmation = input("\nDo you approve this tool call? (YES/no): ")
                if confirmation.lower() == 'yes' or confirmation == "":
                    tool_result = execute_tool(tool_call.function.name, tool_call.function.arguments)
                    tool_result = f"Successfully executed {tool_call.function.name}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                    messages.append({
                        "role": "user",
                        "content": "What's next?"
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": "The tool was not approved and thus did not execute"
                    })
                    continue

            if tool_call.function.name == "inference":
                print("The model is ready to be used. My job here is done. Goodbye!")
                return

            response = get_completion(
                api="openai",
                messages=messages,
                tools=tools,
            )
        
        messages.append({"role": "assistant", "content": response.content})
