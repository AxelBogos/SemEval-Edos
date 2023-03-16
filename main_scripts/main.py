import logging

from dotenv import load_dotenv

from src.utils import defines, setup_logging


def main():
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    setup_logging.setup_python_logging()

    load_dotenv(defines.DOTENV_FILE)
