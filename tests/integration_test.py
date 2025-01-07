import os

ENTITIES = '"Person" "Organization" "Location"'
MISTRAL_API = "mistral"
OPENAI_API = "openai"
NB_SAMPLES = 50
STEPS_ALL = "all"
STEPS_GENERATION = "generation"
STEPS_TRAINING = "training"
STEPS_INFERENCE = "inference"

os.chdir(os.getcwd())


def test_generation_sh():
    path = "test_dataset_1.txt"
    output_dir = "test_ner_model_1"
    os.system(f"uv run -m src.main -e {ENTITIES} -s {STEPS_GENERATION} -a {MISTRAL_API} -nb {
              NB_SAMPLES} -p {path} -od {output_dir}")


def test_training_sh():
    path = "test_dataset_2.txt"
    output_dir = "test_ner_model_2"
    os.system(f"uv run -m src.main -e {ENTITIES} -s {STEPS_TRAINING} -a {MISTRAL_API} -nb {
              NB_SAMPLES} -p {path} -od {output_dir}")


def test_inference_e2e():
    path = "test_dataset_3.txt"
    output_dir = "test_ner_model_3"
    os.system(f"uv run -m src.main -e {ENTITIES} -s {STEPS_INFERENCE} -a {MISTRAL_API} -nb {
              NB_SAMPLES} -p {path} -od {output_dir}")


def test_e2e_sh():
    path = "test_dataset_4.txt"
    output_dir = "test_ner_model_4"
    os.system(f"uv run -m src.main -e {ENTITIES} -s {STEPS_ALL} -a {MISTRAL_API} -nb {
              NB_SAMPLES} -p {path} -od {output_dir}")
