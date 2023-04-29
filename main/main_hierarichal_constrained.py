import logging
import os
import pprint
from argparse import Namespace
from pathlib import Path

import wandb
from dotenv import load_dotenv
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger

from src.models.hierarichal_constrained_transformer_module import (
    HierarchicalTransformerModule,
)
from src.models.transformer_module import TransformerModule
from src.utils import defines, helpers


def main(model_name: str):
    """The main function is the entry point of the program.

    It initializes all necessary variables and objects, then starts training.
    """
    assert model_name in defines.VALID_MODEL_CHOICES, f"Invalid model name: {model_name}"
    # Load environment variables
    load_dotenv(defines.DOTENV_FILE)

    # Setup logging & parse args
    log_dir_path_a, log_dir_path_b, log_dir_path_c = get_log_dirs()

    args_task_a, args_task_b, args_task_c = get_task_args(
        log_dir_path_a, log_dir_path_b, log_dir_path_c, model_name
    )

    # Build data module
    data_module_task_a = helpers.get_data_module(args_task_a)
    data_module_task_b = helpers.get_data_module(args_task_b)
    data_module_task_c = helpers.get_data_module(args_task_c)

    data_module_task_a.setup()
    data_module_task_b.setup()
    data_module_task_c.setup()

    args_task_a.num_target_class = data_module_task_a._num_classes
    args_task_a.len_train_loader = len(data_module_task_a.train_dataloader())
    args_task_b.num_target_class = data_module_task_b._num_classes
    args_task_b.len_train_loader = len(data_module_task_b.train_dataloader())
    args_task_c.num_target_class = data_module_task_c._num_classes
    args_task_c.len_train_loader = len(data_module_task_c.train_dataloader())

    seed_everything(args_task_a.random_seed)

    download_links = helpers.get_model_download_links(model_name)
    model_paths = list()
    api = wandb.Api()
    for link in download_links:
        artifact = api.artifact(link)
        model_paths.append(artifact.download() + "/model.ckpt")

    wrapper_model_task_a, wrapper_model_task_b, wrapper_model_task_c = get_wrapper_models(
        args_task_a, args_task_b, args_task_c, model_name, model_paths
    )

    # Task A
    if args_task_a.train:
        # Setup WandB logging
        wandb_logger_a = WandbLogger(
            project="EDOS-ift6289",
            save_dir=log_dir_path_a,
            log_model=True,
            group="Task a",
            tags=[model_name, "hierarchical"],
        )
        lightning_callbacks_a = [
            ModelSummary(),
            ModelCheckpoint(dirpath=log_dir_path_a, monitor="val/f1", save_top_k=1, mode="max"),
            EarlyStopping(monitor="val/loss", patience=args_task_a.patience),
        ]
        trainer_a = Trainer(
            logger=wandb_logger_a,
            callbacks=lightning_callbacks_a,
            accelerator="auto",
            devices="auto",
            max_epochs=args_task_a.num_epoch,
        )
        trainer_a.fit(model=wrapper_model_task_a, datamodule=data_module_task_a)

    if args_task_a.eval:
        trainer_a.test(model=wrapper_model_task_a, datamodule=data_module_task_a, ckpt_path="best")
    wandb.finish()

    # Task B
    if args_task_b.train:
        wandb_logger_b = WandbLogger(
            project="EDOS-ift6289",
            save_dir=log_dir_path_b,
            log_model=True,
            group="Task b",
            tags=[model_name, "hierarchical"],
        )
        lightning_callbacks_b = [
            ModelSummary(),
            ModelCheckpoint(dirpath=log_dir_path_b, monitor="val/f1", save_top_k=1, mode="max"),
            EarlyStopping(monitor="val/loss", patience=args_task_b.patience),
        ]
        trainer_b = Trainer(
            logger=wandb_logger_b,
            callbacks=lightning_callbacks_b,
            accelerator="auto",
            devices="auto",
            max_epochs=args_task_b.num_epoch,
        )
        trainer_b.fit(model=wrapper_model_task_b, datamodule=data_module_task_b)
    if args_task_b.eval:
        trainer_b.test(model=wrapper_model_task_b, datamodule=data_module_task_b, ckpt_path="best")
    wandb.finish()
    # Task C
    if args_task_c.train:
        wandb_logger_c = WandbLogger(
            project="EDOS-ift6289",
            save_dir=log_dir_path_c,
            log_model=True,
            group="Task c",
            tags=[model_name, "hierarchical"],
        )
        lightning_callbacks_c = [
            ModelSummary(),
            ModelCheckpoint(dirpath=log_dir_path_c, monitor="val/f1", save_top_k=1, mode="max"),
            EarlyStopping(monitor="val/loss", patience=args_task_c.patience),
        ]
        trainer_c = Trainer(
            logger=wandb_logger_c,
            callbacks=lightning_callbacks_c,
            accelerator="auto",
            devices="auto",
            max_epochs=args_task_c.num_epoch,
        )
        trainer_c.fit(model=wrapper_model_task_c, datamodule=data_module_task_c)
    if args_task_c.eval:
        trainer_c.test(model=wrapper_model_task_c, datamodule=data_module_task_c, ckpt_path="best")


def get_wrapper_models(args_task_a, args_task_b, args_task_c, model_name, model_paths):
    classifier_a = TransformerModule.load_from_checkpoint(model_paths[0])
    classifier_b = TransformerModule.load_from_checkpoint(model_paths[1])
    classifier_c = TransformerModule.load_from_checkpoint(model_paths[2])
    wrapper_model_task_a = HierarchicalTransformerModule(
        model=model_name,
        learning_rate=args_task_a.lr,
        classifier_a=classifier_a,
        classifier_b=classifier_b,
        classifier_c=classifier_c,
        task="a",
    )
    wrapper_model_task_b = HierarchicalTransformerModule(
        model=model_name,
        learning_rate=args_task_b.lr,
        classifier_a=classifier_a,
        classifier_b=classifier_b,
        classifier_c=classifier_c,
        task="b",
    )
    wrapper_model_task_c = HierarchicalTransformerModule(
        model=model_name,
        learning_rate=args_task_c.lr,
        classifier_a=classifier_a,
        classifier_b=classifier_b,
        classifier_c=classifier_c,
        task="c",
    )
    return wrapper_model_task_a, wrapper_model_task_b, wrapper_model_task_c


def get_log_dirs():
    if not os.path.isdir(defines.LOG_DIR):
        os.mkdir(defines.LOG_DIR)
    log_dir_path_a = Path(defines.LOG_DIR, helpers.get_time() + "a")
    os.mkdir(log_dir_path_a)
    log_dir_path_b = Path(defines.LOG_DIR, helpers.get_time() + "b")
    os.mkdir(log_dir_path_b)
    log_dir_path_c = Path(defines.LOG_DIR, helpers.get_time() + "c")
    os.mkdir(log_dir_path_c)
    helpers.setup_python_logging(log_dir_path_a)
    return log_dir_path_a, log_dir_path_b, log_dir_path_c


def get_task_args(log_dir_path_a, log_dir_path_b, log_dir_path_c, model_name):
    args_task_a = {
        "train": True,
        "eval": True,
        "task": "a",
        "architecture": "transformer",
        "model": model_name,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "lr": 5e-06,
        "step_scheduler": 5,
        "n_warmup_steps": 0,
        "preprocessing_mode": "none",
        "max_token_length": 128,
        "batch_size": 16,
        "num_epoch": 9,
        "patience": 3,
        "num_workers": 1,
        "random_seed": 3454572,
        "logs_dir": defines.LOG_DIR,
        "raw_data_dir": defines.RAW_DATA_DIR,
        "interim_data_dir": defines.INTERIM_DATA_DIR,
        "processed_data_dir": defines.PROCESSED_DATA_DIR,
        "log_dir": log_dir_path_a,
    }
    args_task_b = args_task_a.copy()
    args_task_c = args_task_a.copy()
    args_task_b["task"] = "b"
    args_task_c["task"] = "c"
    args_task_b["log_dir"] = log_dir_path_b
    args_task_c["log_dir"] = log_dir_path_c
    args_task_a = Namespace(**args_task_a)
    args_task_b = Namespace(**args_task_b)
    args_task_c = Namespace(**args_task_c)
    return args_task_a, args_task_b, args_task_c


if __name__ == "__main__":
    main("roberta-base")
