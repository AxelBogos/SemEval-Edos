import logging
import os
from datetime import datetime
from pathlib import Path

from src.utils import defines


def setup_python_logging(log_dir: Path = None) -> None:
    """The setup_python_logging function configures the Python logging module to log messages to a
    file and also to the console.  The function takes an optional argument, log_dir, which is a
    Path object pointing to where you want your logs saved.  If no value is passed for this
    argument then only console logging will be enabled.

    :param log_dir: Path: Specify the directory where the logs will be stored
    :return: Nothing
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if log_dir is None:
        # log to console
        logging.basicConfig(level=logging.INFO, format=log_fmt)
    else:
        # log to file
        logging.basicConfig(level=logging.INFO, format=log_fmt, filename=Path(log_dir, "logs.txt"))

        # log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)


def setup_wandb():
    pass


def _get_time():
    """The _get_time function returns the current time in a string format.

    :return: A string with the current time in this format: YYYY-MM-DD-HH-MM-SS
    """
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def make_log_dir() -> Path:
    """The make_log_dir function creates a directory in the log directory with the current time as
    its name. The function returns None if there is already a folder with that name, and otherwise
    returns the path to this new folder.

    :return: A path to a new directory
    """
    log_dir_path = Path(defines.LOG_DIR, _get_time())
    os.mkdir(log_dir_path)
    return log_dir_path


def get_model(args):
    pass
