import logging


def setup_python_logging() -> None:
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)


def setup_wandb():
    pass
