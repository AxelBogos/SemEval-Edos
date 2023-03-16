import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.utils import defines, raw_data_downloader, setup_logging


def main():
    setup_logging.setup_python_logging()
    logger = logging.getLogger(__name__)

    # Download data
    logger.info("Downloading raw data...")
    downloader = raw_data_downloader.GoogleDriveDownloader(defines.DATA_DIR)
    downloader.download()

    # Process data


if __name__ == "__main__":
    main()
