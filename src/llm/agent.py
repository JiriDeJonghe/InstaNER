import json
from typing import Any

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
                        "description": "Identifier of API. Must be 'mistral', 'openai', or 'gemini'."
                    },
                    "nb_samples": {
                        "type": "integer",
                        "description": "Number of samples that need to be generated for the dataset"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the generated dataset"
                    },
                    "examples": {
                        "type": "array",
                        "description": "Examples on which the generated samples can be based on",
                        "items": {
                            "type": "string"
                        }
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

You are given a set of tools that you can use to achieve this task. 
- Always asks the user for ALL of the required arguments of the functions, if they don't provide any, propose something and explain why. 
- Always asks whether the user wants to add optional arguments.
- Only start calling tools when you have all the requirements that you need.
- Only provide suggestions if the user explicitely asks for it
- Do not make assumptions regarding the input unless explicitely stated in the function description
"""


tool_registry = {
    "generate_dataset": generate_dataset,
    "train_model": train_model,
    "evaluate_model": evaluate_model,
    "inference": inference
}

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Executes the tool with the given arguments

    Args:
        tool_name (str): name of the tool to execute
        arguments (dict[str, Any]): arguments to call the tool with

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
            api="gemini",
            messages=messages,
            tools=tools,
        )

        if response.function_call:
            assistant_message = {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {
                        "name": response.function_call.name,
                        "arguments": response.function_call.args
                    }
                ]
            }
            messages.append(assistant_message)

            print("\nAssistant:", response.text)
            print(f"\nI would like to call the following tool:")
            print(f"Tool: {response.function_call.name}")
            print(f"Arguments: {response.function_call.args}")
            
            tool_result = execute_tool(response.function_call.name, response.function_call.args)
            tool_result = f"Successfully executed {response.function_call.name}"
            messages.append({
                "role": "tool",
                "name": response.function_call.name,
                "content": tool_result
            })
            messages.append({
                "role": "user",
                "content": "What's next?"
            })

            if response.function_call.name == "inference":
                print("The model is ready to be used. My job here is done. Goodbye!")
                return

            response = get_completion(
                api="gemini",
                messages=messages,
                tools=tools,
            )
        
        messages.append({"role": "assistant", "content": response.text})
