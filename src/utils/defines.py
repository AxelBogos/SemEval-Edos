from pathlib import Path

import pyrootutils

ROOT_PATH = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
DATA_DIR = Path(ROOT_PATH, "data")
DOTENV_FILE = Path(ROOT_PATH, ".env")
LOG_DIR = Path(ROOT_PATH, "logs")
