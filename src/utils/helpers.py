import logging
import os
from datetime import datetime
from pathlib import Path

import torch.optim
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.data import edos_datamodule
from src.models import lstm_module
from src.models.components import simple_bilstm_net
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
        formatter = logging.Formatter(log_fmt)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)


def setup_wandb(args):
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        save_dir=args.log_dir,
        log_model=True,
        group=args.model,
        tags=args.model,
    )
    return wandb_logger


def get_lightning_callbacks(args):
    callbacks = list()
    callbacks.append(ModelSummary())
    callbacks.append(
        ModelCheckpoint(dirpath=args.log_dir, monitor="val/loss", save_top_k=1, mode="min")
    )
    callbacks.append(EarlyStopping(monitor="val/loss", patience=args.patience))
    return callbacks


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


def get_model(
    args, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None
):
    if args.model == "bilstm":
        net = simple_bilstm_net.SimpleBiLstmNet(args)
        return lstm_module.LSTMModule(net, optimizer, scheduler)
    if args.model == "gnb":
        pass  # TODO
    if args.model == "distillbert":
        pass  # TODO


def get_data_module(args):
    datamodule = edos_datamodule.EDOSDataModule(args)
    return datamodule


def get_optimizer(args):
    """The get_optimizer function takes in the args object and returns an optimizer.

    :param args: Pass in the arguments from the command line
    :return: An optimizer
    """
    if args.optimizer == "Adam":
        optimizer = optim.Adam
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW
    if args.optimizer == "SGD":
        optimizer = optim.SGD

    return optimizer


def get_scheduler(args):
    return None  # TODO
