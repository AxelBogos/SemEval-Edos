import logging

import src.utils.helpers
from src.utils import defines, interim_preprocessing, raw_data_downloader, setup_logging


def main():
    src.utils.helpers.setup_python_logging()
    logger = logging.getLogger(__name__)

    # Download data
    logger.info("Downloading raw data...")
    downloader = raw_data_downloader.GoogleDriveDownloader(defines.RAW_DATA_DIR)
    downloader.download()

    # Encode and merge dev & test set labels
    interim_processor = interim_preprocessing.InterimProcessor(
        defines.RAW_DATA_DIR, defines.INTERIM_DATA_DIR
    )
    interim_processor.run()


if __name__ == "__main__":
    main()
