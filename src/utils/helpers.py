import logging
import os
from datetime import datetime
from pathlib import Path

from src.utils import defines


def setup_python_logging() -> None:
    """The setup_python_logging function sets up the Python logging module to log messages to
    stdout. The function is called by main() before any other code in this file is executed.

    :return: None
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


def _setup_wandb():
    pass


def _get_time():
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def _make_log_dir() -> bool:
    logger = logging.getLogger(__name__)
    log_dir_path = Path(defines.LOG_DIR, _get_time())
    if log_dir_path.is_dir():
        logger.critical("Run log directory already exists")
        return False
    os.mkdir(log_dir_path)
    return True
