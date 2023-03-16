import logging


def _setup_python_logging() -> None:

    """The _setup_python_logging function sets up the Python logging module to log messages to
    stdout. The function is called by main() before any other code in this file is executed.

    :return: None
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


def _setup_wandb():
    pass
