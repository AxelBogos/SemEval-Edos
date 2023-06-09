import logging
import os
from datetime import datetime
from pathlib import Path

import torch.optim
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

from src.data import datamodule_lstm, datamodule_transformer
from src.models import lstm_module, transformer_module
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
        project="EDOS-ift6289",
        save_dir=args.log_dir,
        log_model=True,
        group="Task " + args.task,
        tags=[args.model],
    )
    return wandb_logger


def get_lightning_callbacks(
    log_dir: Path,
    model_checkpoint_monitor: str,
    model_checkpoint_mode: str,
    early_stopping_patience: int,
):
    """The get_lightning_callbacks function returns a list of callbacks that can be used in the
    Trainer.

    :param log_dir:Path: Define the directory where the model checkpoints will be saved
    :param model_checkpoint_monitor:str: Monitor the metric that we want to use for saving the model
    :param model_checkpoint_mode:str: Determine how the model is saved
    :param early_stopping_patience:int: Determine how many epochs to wait before stopping the training
    :return: A list of callbacks
    """
    callbacks = list()
    callbacks.append(ModelSummary())
    callbacks.append(
        ModelCheckpoint(
            dirpath=log_dir,
            monitor=model_checkpoint_monitor,
            save_top_k=1,
            mode=model_checkpoint_mode,
        )
    )
    callbacks.append(EarlyStopping(monitor="val/loss", patience=early_stopping_patience))
    return callbacks


def get_time():
    """The get_time function returns the current time in a string format.

    :return: A string with the current time in this format: YYYY-MM-DD-HH-MM-SS
    """
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def make_log_dir() -> Path:
    """The make_log_dir function creates a directory in the log directory with the current time as
    its name. The function returns None if there is already a folder with that name, and otherwise
    returns the path to this new folder.

    :return: A path to a new directory
    """
    log_dir_path = Path(defines.LOG_DIR, get_time())
    if not defines.LOG_DIR.is_dir():
        os.mkdir(defines.LOG_DIR)
    os.mkdir(log_dir_path)
    return log_dir_path


def get_model(args, optimizer: torch.optim.Optimizer = None):
    """The get_model function is a factory function that returns an instance of the nn.Module
    class. The get_model function takes in two optional arguments: optimizer and scheduler. These
    are used to pass in PyTorch objects that will be used to train our model.

    :param args: Pass arguments to the model
    :param optimizer: torch.optim.Optimizer:
    :param scheduler: torch.optim.lr_scheduler:
    :return: A model object that is a torch.nn.Module
    """
    if args.architecture == "lstm":
        return lstm_module.LSTMModule(args=args, optimizer=optimizer)
    elif args.architecture == "transformer":
        return transformer_module.TransformerModule(
            model=args.model,
            num_target_class=args.num_target_class,
            learning_rate=args.lr,
            num_epoch=args.num_epoch,
            n_warmup_steps=args.n_warmup_steps,
            len_train_loader=args.len_train_loader,
            optimizer=optimizer,
        )


def get_data_module(args):
    """The get_data_module function is used to create a data module object. The data module object
    is responsible for loading the dataset and creating the dataloaders. It also contains any other
    functions that are needed to process the dataset, such as normalization or augmentation.

    :param args: Pass in the arguments from the command line
    :return: The data module
    """
    if args.architecture == "lstm":
        return datamodule_lstm.DataModuleLSTM(args)
    elif args.architecture == "transformer":
        return datamodule_transformer.DataModuleTransformer(args)


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


def get_model_download_links(model_name):
    if model_name == "distilroberta-base":
        model_a = "axel-bogos/EDOS-ift6289/model-t7y8oy8u:v0"
        model_b = "axel-bogos/EDOS-ift6289/model-g5w2beck:v0"
        model_c = "axel-bogos/EDOS-ift6289/model-eu1j2viv:v0"
    elif model_name == "roberta-base":
        model_a = "axel-bogos/EDOS-ift6289/model-rn477e90:v0"
        model_b = "axel-bogos/EDOS-ift6289/model-q4twdy6m:v0"
        model_c = "axel-bogos/EDOS-ift6289/model-hsf7y76o:v0"
    else:
        raise ValueError("Invalid model name")
    return model_a, model_b, model_c
