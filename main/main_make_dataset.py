import logging

from src.utils import defines, helpers, interim_preprocessing, raw_data_downloader


def main():
    helpers.make_data_dirs()
    helpers.setup_python_logging()
    logger = logging.getLogger(__name__)

    # Download data
    logger.info("Downloading raw data...")
    downloader = raw_data_downloader.GoogleDriveDownloader(defines.RAW_DATA_DIR)
    downloader.download()

    # Encode and merge val & test set labels
    interim_processor = interim_preprocessing.InterimProcessor(
        defines.RAW_DATA_DIR, defines.INTERIM_DATA_DIR
    )
    interim_processor.run()


if __name__ == "__main__":
    main()
