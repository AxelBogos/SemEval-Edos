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


def make_data_dirs() -> None:
    """The make_data_dirs function creates the data directories if they do not already exist.

    :return: None
    """
    if not defines.DATA_DIR.is_dir():
        os.mkdir(defines.DATA_DIR)
    if not defines.RAW_DATA_DIR.is_dir():
        os.mkdir(defines.RAW_DATA_DIR)
    if not defines.INTERIM_DATA_DIR.is_dir():
        os.mkdir(defines.INTERIM_DATA_DIR)
    if not defines.PROCESSED_DATA_DIR.is_dir():
        os.mkdir(defines.PROCESSED_DATA_DIR)
    if not defines.EXTERNAL_DATA_DIR.is_dir():
        os.mkdir(defines.EXTERNAL_DATA_DIR)


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
        if not log_dir.is_dir():
            os.mkdir(log_dir)
        logging.basicConfig(level=logging.INFO, format=log_fmt, filename=Path(log_dir, "logs.txt"))

        # log to console
        console = logging.StreamHandler()
        formatter = logging.Formatter(log_fmt)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)


def setup_wandb(args):
    """The setup_wandb function is used to initialize the wandb logger.

    :param args: Pass in the command line arguments
    :return: A wandblogger object
    """
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        save_dir=args.log_dir,
        log_model=True,
        group=args.model,
        tags=args.model,
    )
    return wandb_logger


def get_lightning_callbacks(args):
    """The get_lightning_callbacks function returns a list of callbacks that are used by the
    LightningModule. The ModelSummary callback prints out the model summary to stdout. The
    ModelCheckpoint callback saves checkpoints to disk, and only keeps the best one based on
    validation loss. The EarlyStopping callback stops training if validation loss does not improve
    after a certain number of epochs.

    :param args: Pass in the arguments from the command line
    :return: A list of callbacks
    """
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
    """The get_model function is a factory function that returns an instance of the nn.Module
    class. The get_model function takes in two optional arguments: optimizer and scheduler. These
    are used to pass in PyTorch objects that will be used to train our model.

    :param args: Pass arguments to the model
    :param optimizer: torch.optim.Optimizer:
    :param scheduler: torch.optim.lr_scheduler:
    :return: A model object that is a torch.nn.Module
    """
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
