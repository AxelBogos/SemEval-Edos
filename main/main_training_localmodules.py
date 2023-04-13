import logging
import pprint
from argparse import Namespace

import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from src.data.datamodule_transformer_local import DataModuleTransformerLocal
from src.models.transformer_module_local import TransformerModuleLocal
from src.utils import defines, helpers


def main(model_name: str):
    """The main function is the entry point of the program.

    It initializes all necessary variables and objects, then starts training.
    """
    assert model_name in defines.VALID_MODEL_CHOICES, f"Invalid model name: {model_name}"
    # Load environment variables
    load_dotenv(defines.DOTENV_FILE)

    # Setup logging & parse args
    log_dir = helpers.make_log_dir()
    helpers.setup_python_logging(log_dir)

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
        "log_dir": log_dir,
    }
    args_task_b = args_task_a.copy()
    args_task_c1 = args_task_a.copy()
    args_task_c2 = args_task_a.copy()
    args_task_c3 = args_task_a.copy()
    args_task_c4 = args_task_a.copy()
    args_task_b["task"] = "b"
    args_task_c1["task"] = "c"
    args_task_c2["task"] = "c"
    args_task_c3["task"] = "c"
    args_task_c4["task"] = "c"
    args_task_a = Namespace(**args_task_a)
    args_task_b = Namespace(**args_task_b)
    args_task_c1 = Namespace(**args_task_c1)
    args_task_c2 = Namespace(**args_task_c2)
    args_task_c3 = Namespace(**args_task_c3)
    args_task_c4 = Namespace(**args_task_c4)
    args_task_a.log_dir = log_dir
    args_wrapper = {
        "train": True,
        "eval": True,
        "task": "multitask",
        "architecture": "transformer-wrapper",
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
        "log_dir": log_dir,
    }
    args_wrapper = Namespace(**args_wrapper)

    # Setup WandB logging
    wandb_logger = WandbLogger(
        project="EDOS-ift6289",
        save_dir=log_dir,
        log_model=True,
        group="Local Clf Multitask",
        tags=[model_name],
    )

    logger = logging.getLogger(__name__)
    seed_everything(args_task_a.random_seed)

    # Build data module
    data_module_task_a = DataModuleTransformerLocal(subtask="a")
    data_module_task_b = DataModuleTransformerLocal(subtask="b")
    data_module_task_c1 = DataModuleTransformerLocal(subtask="c1")
    data_module_task_c2 = DataModuleTransformerLocal(subtask="c2")
    data_module_task_c3 = DataModuleTransformerLocal(subtask="c3")
    data_module_task_c4 = DataModuleTransformerLocal(subtask="c4")

    data_module_task_a.setup()
    data_module_task_b.setup()
    data_module_task_c1.setup()
    data_module_task_c2.setup()
    data_module_task_c3.setup()
    data_module_task_c4.setup()

    args_task_a.num_target_class = data_module_task_a._num_classes
    args_task_a.len_train_loader = len(data_module_task_a.train_dataloader())
    args_task_b.num_target_class = data_module_task_b._num_classes
    args_task_b.len_train_loader = len(data_module_task_b.train_dataloader())
    args_task_c1.num_target_class = data_module_task_c1._num_classes
    args_task_c1.len_train_loader = len(data_module_task_c1.train_dataloader())
    args_task_c2.num_target_class = data_module_task_c2._num_classes
    args_task_c2.len_train_loader = len(data_module_task_c2.train_dataloader())
    args_task_c3.num_target_class = data_module_task_c3._num_classes
    args_task_c3.len_train_loader = len(data_module_task_c3.train_dataloader())
    args_task_c4.num_target_class = data_module_task_c4._num_classes
    args_task_c4.len_train_loader = len(data_module_task_c4.train_dataloader())
    # Build models
    model_task_a = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_a",
        num_target_class=data_module_task_a._num_classes,
        len_train_loader=len(data_module_task_a.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )
    model_task_b = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_b",
        num_target_class=data_module_task_b._num_classes,
        len_train_loader=len(data_module_task_b.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )
    model_task_c1 = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_c1",
        num_target_class=data_module_task_c1._num_classes,
        len_train_loader=len(data_module_task_c1.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )
    model_task_c2 = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_c2",
        num_target_class=data_module_task_c2._num_classes,
        len_train_loader=len(data_module_task_c2.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )
    model_task_c3 = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_c3",
        num_target_class=data_module_task_c3._num_classes,
        len_train_loader=len(data_module_task_c3.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )
    model_task_c4 = TransformerModuleLocal(
        model=model_name,
        subtask="subtask_c4",
        num_target_class=data_module_task_c4._num_classes,
        len_train_loader=len(data_module_task_c4.train_dataloader()),
        num_epoch=9,
        learning_rate=5e-06,
        optimizer=torch.optim.AdamW,
    )

    lightning_callbacks = helpers.get_lightning_callbacks(args_task_a)

    trainer_a = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_a.num_epoch,
    )
    trainer_b = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_b.num_epoch,
    )
    trainer_c1 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_c1.num_epoch,
    )
    trainer_c2 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_c2.num_epoch,
    )
    trainer_c3 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_c3.num_epoch,
    )
    trainer_c4 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_c4.num_epoch,
    )

    # Train
    trainer_a.fit(model_task_a, datamodule=data_module_task_a)
    trainer_b.fit(model_task_b, datamodule=data_module_task_b)
    trainer_c1.fit(model_task_c1, datamodule=data_module_task_c1)
    trainer_c2.fit(model_task_c2, datamodule=data_module_task_c2)
    trainer_c3.fit(model_task_c3, datamodule=data_module_task_c3)
    trainer_c4.fit(model_task_c4, datamodule=data_module_task_c4)


if __name__ == "__main__":
    main("distilroberta-base")
