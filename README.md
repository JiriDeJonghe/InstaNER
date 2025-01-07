# InstaNER

InstaNER is a CLI tool that automates the Named Entity Recognition (NER) model training workflow by automatically generating training datasets using Large Language Models and fine-tuning a transformer model for NER tasks.

The goal of this project is to provide an easy-to-use accelerator allowing the user to train a new NER model for any entities specified.

## Features
- Automatic dataset generation using state-of-the-art LLMs (currently supporting OpenAI and Mistral)
- Customizing the final trained model by defining the entities that should be recognized, language, and other parameters
- Easy fine-tuning of a locally saved model using Transformers and PyTorch
- Flexible use of custom/local LLMs by simply defining one function call

To create a NER model from scratch, two steps are done sequentially:
1. **Creation of synthetic data**: first synthetic data is created by making a series of calls to your preferred LLM. The prompt is set up in such a way that it will provide examples for each entities that you define that should be covered. The created data will be saved in a .txt file in the `data` directory.
2. **Training the NER model**: using the synthetic data, in the next step a NER model is trained using the Transformers library. The final model will be saved in the `models` directory.

The synthetic data will automatically be separated in a train- and test-dataset. The fine-tuned model will be evaluated on the test-dataset. Finally, the model will be loaded and can be used for real-time inference using the CLI.

## Getting Started

### Requirements
- (If wanting to use OpenAI): **OpenAI API key**: this tool can use the OpenAI API to create synthetic data
- (If wanting to use Mistral): **Mistral key**: this tool can use the OpenAI API to create synthetic data
- **[uv](https://github.com/astral-sh/uv) installation**: this is not strictly required, but will make your life easier

### Installing
Installing the requirements is straightforward using the ```uv``` package manager
1. Navigate to the directory of this repo in your terminal
2. Set up uv to instantiate your virtual environment
3. Use ```uv sync``` to install all the required modules


### API Key Configuration
You'll need to set up your `.env` file and store it in the `src` directory. A template for OpenAI and Mistral fields that are used is provided. I'll leave it to the user if they want to use a different LLM or API. 

### Fastest way to your trained model
The easiest way to use this tool is to simply running the entire pipeline and passing the entities you want your model to recognize. This will generate the synthetic data as well as training a NER model based on this data. It will also load the model for inference and print out the results on a simple example.
1. Navigate to the directory of this repo in your terminal
2. Use ```uv run -m src,main -e "Entity1" "Entity2" "Entity3"```

### Customization
If you want to run only part of the pipeline and/or want more control over the process, you can configure the following parameters:
1. **Entities** (```-e```): a list of entities that should be recognized by the NER model.
2. **Language** (```-l```, optional, default: "english"): the language of the dataset and NER model.
3. **API** (```-a```, optional, default: "mistral"): the API to use for creating synthetic data. Can be either "openai" or "mistral".
4. **nb_samples** (```-nb```, optional, default: 1000): the number of samples to generate for the dataset.
5. **path** (```-p```, optional, default: "dataset.txt"): the path to save the generated dataset.
6. **steps** (```-s```, optional, default: "all"): the steps to run in the pipeline. Can be either "all", "generation", "training", "evaluation", "inference" or a combination of multiple.

You can pass these parameters as arguments in the command line, e.g.,:
```uv run -m src.main -e "Entity1" "Entity2" "Entity3" -l "dutch" -a "openai" -nb 1000 -p "dataset.txt" -s "training" -od "ner_model"```

### Testing
The different pipelines can be tested using the ```pytest``` framework. To do this, navigate to the home directory and run ```uv run python -m pytest```. This will run the tests for the entire repo. If you want to test a specific pipeline, navigate to the directory of the pipeline and run ```uv run python -m pytest tests/[test_file_name].py```.

### Future Improvements
- Improve synthetic data generation by increasing variety of samples
- Add SpaCy integration
- Add testing for CoNLL 2002 for Dutch
- Add support for benchmarking
- Add support for other LLMs
