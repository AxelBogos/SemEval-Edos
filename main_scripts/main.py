import logging

from dotenv import load_dotenv

import src.utils.helpers
from src.utils import defines


def main():
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    src.utils.helpers.setup_python_logging()

    load_dotenv(defines.DOTENV_FILE)
