import logging
from argparse import Namespace

import torch
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

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

    # Setup WandB logging
    wandb_logger = WandbLogger(
        project="EDOS-ift6289",
        save_dir=log_dir,
        log_model=True,
        group="Local Clf Multitask",
        tags=[model_name],
    )

    logger = logging.getLogger(__name__)
    seed_everything(3454572)

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

    lightning_callbacks_a = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_a/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_a/loss", patience=3),
    ]
    lightning_callbacks_b = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_b/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_b/loss", patience=3),
    ]
    lightning_callbacks_c1 = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_c1/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_c1/loss", patience=3),
    ]
    lightning_callbacks_c2 = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_c2/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_c2/loss", patience=3),
    ]
    lightning_callbacks_c3 = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_c3/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_c3/loss", patience=3),
    ]
    lightning_callbacks_c4 = [
        ModelSummary(),
        ModelCheckpoint(dirpath=log_dir, monitor="val_subtask_c4/loss", save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_subtask_c4/loss", patience=3),
    ]

    trainer_a = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_a,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
    )
    trainer_b = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_b,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
    )
    trainer_c1 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_c1,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
    )
    trainer_c2 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_c2,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
    )
    trainer_c3 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_c3,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
    )
    trainer_c4 = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks_c4,
        accelerator="auto",
        devices="auto",
        max_epochs=9,
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
