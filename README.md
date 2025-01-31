# InstaNER

InstaNER is a CLI tool that automates the entire process of creating a Named Entity Recognition (NER) model, starting all the way from identifying the needed entities, creating train and test data, training a transformer model on the data, evaluating the results and loading it for inference, all while ensuring reproducability.

The goal of this project is to provide an easy-to-use accelerator allowing the user to train a new NER model for any entities specified. The accelerator makes use of an LLM so you either have to provide your own API keys or connect to a locally running model.

There's two ways to use the tool:
- Using the CLI to start the `main` function with arguments. This approach is recommended if you know what you're doing and have trained NER models before.
- Using the Model Creation Agent, which uses an agentic workflow that will guide you through the process of generating synthetic data, as well as training and evaluating the model. This makes the tool accessible to users that are not too familiar with the training of an AI model.

In both cases, you will end up with a fully trained NER model that can be used. All the generated data and training hyperparameters will be put in the `./experiments/{language}/{save_dir}` for easy reusability and reproducability.

## Features
- Easily train a NER model from scratch going through the entire process of data gathering, training and evaluating
- Automatic dataset generation using state-of-the-art LLMs (currently supporting OpenAI, Mistral and DeepSeek - or any other API that can be called via OpenAI)
- Customizing the final trained model by defining the entities that should be recognized, the language of the final model, and other hyperparameters (e.g., number of samples to train on)
- Integration of the Model Creation Agent, allowing users without any previous ML experience to train a model
- Easy fine-tuning of a locally saved model using Transformers and PyTorch
- Flexible use of custom/local LLMs by simply configuring one function call
- Easy DevOps integration by keeping track of all the training arguments, evaluations of the model and everything needed to reproduce the training of the model

To create a NER model from scratch, two steps are done sequentially:
1. **Creation of synthetic data**: first synthetic data is created by making a series of calls to your preferred LLM. The prompt is set up in such a way that it will provide examples for each entities that should be covered according to your specifications. The created data will be saved in a newly created directory that will be configured according to your language and, if supplied, save directory. If a directory is not specifically defined, a timestamp of moment of creation will be used instead
2. **Training the NER model**: using the synthetic data, in the next step a NER model is trained using the Transformers library. The final model will be saved in the same directory that was created for this specific experiment.

The synthetic data will automatically be separated in a train- and test-dataset. The fine-tuned model will be evaluated on the test-dataset. Finally, the model will be loaded and can be used for real-time inference using the CLI. The train- and test-dataset and other parameters can be found in the `config.json` file.

## Getting Started

### Requirements
- (If wanting to use OpenAI or other OpenAI based API, e.g., DeepSeek): **API key**: this tool can use the OpenAI API to create synthetic data.
- (If wanting to use Mistral): **Mistral key**: this tool can use the Mistral API to create synthetic data. Mistral provides a free to use API that can be used as an alternative option.
- **Python**: the tool is ran using Python, so a Python installation is required, recommended latest one.
- **[uv](https://github.com/astral-sh/uv) installation**: this is not strictly required, but will make your life easier, and therefore is strongly recommended.

### Installing
Installing the needed packages is very straightforward using the ```uv``` package manager
1. Navigate to the directory of this repo in your terminal
2. Set up `uv` to instantiate your virtual environment
3. Use ```uv sync``` to install all the required modules


### API Key Configuration
You'll need to set up your `.env` file and store it in the `src` directory. A template for OpenAI and Mistral fields that are used is provided. I'll leave it to the user if they want to use a different LLM or API. 

### Fastest way to your trained model
1. **Using the agent**: the easiest way to use this tool is to simply start running the agent. This can be done by first navigating in your terminal to the directory where this repo is saved. After that, run the following command ```uv run -m src.main -a True```. This will instantiate the agent which will guide you through the process of training your own NER model. Note that using the agent incurs extra costs as opposed to simply running the pipeline
2. **Running the complete pipeline**: if you don't feel like using the agent or you already know how the tool works and want to simply run the pipeline instead, you can simply run ```uv run -m src.main -e "Entity1" "Entity2"```. This pipeline will create data, train the mode, evaluate the model and load it for inference.

### Customization
If you want to run only part of the pipeline and/or want more control over the process, you can configure the following parameters:
1. **Entities** (```-e```): a list of entities that should be recognized by the NER model.
2. **Language** (```-l```, optional, default: "english"): the language of the dataset and NER model.
3. **API** (```-ap```, optional, default: "mistral"): the API to use for creating synthetic data. Can be either "openai" or "mistral".
4. **nb_samples** (```-nb```, optional, default: 1000): the number of samples to generate for the dataset.
5. **path** (```-p```, optional, default: "dataset.txt"): the path to save the generated dataset.
6. **steps** (```-s```, optional, default: "all"): the steps to run in the pipeline. Can be either "all", "generation", "training", "evaluation", "inference" or a combination of multiple.
7. **agent**: (```-a```): boolean indicating whether to start the agent or to use the normal CLI instead

You can pass these parameters as arguments in the command line, e.g.,:
```uv run -m src.main -e "Entity1" "Entity2" "Entity3" -l "dutch" -a "openai" -nb 1000 -p "dutch_experiment" -s "training"```

### Testing
The different pipelines can be tested using the ```pytest``` framework. To do this, navigate to the home directory and run ```uv run python -m pytest```. This will run the tests for the entire repo. If you want to test a specific pipeline, navigate to the directory of the pipeline and run ```uv run python -m pytest tests/[test_file_name].py```.

### Future Improvements
- Improve synthetic data generation by increasing variety of samples
- Add SpaCy integration as alternative to transformer based architectures
- Add testing for CoNLL 2002 for Dutch
- Add support for benchmarking
- Add support for other LLMs
