from pathlib import Path

import pyrootutils

ROOT_PATH = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
DATA_DIR = Path(ROOT_PATH, "data")
EXTERNAL_DATA_DIR = Path(ROOT_PATH, "data", "external")
INTERIM_DATA_DIR = Path(ROOT_PATH, "data", "interim")
PROCESSED_DATA_DIR = Path(ROOT_PATH, "data", "processed")
RAW_DATA_DIR = Path(ROOT_PATH, "data", "raw")
DOTENV_FILE = Path(ROOT_PATH, ".env")
LOG_DIR = Path(ROOT_PATH, "logs")
VALID_MODEL_CHOICES = ("gnb", "bilstm", "distillbert")
VALID_TASK_CHOICES = ("a", "b", "c")
VALID_PREPROCESSING_MODE = ("standard", "none")
VALID_OPTIMIZERS = ("Adam", "AdamW", "SGD")
