import logging
import pprint
from argparse import Namespace

import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything

from src.models.wrapper_transformer_module import WrapperTransformerModule
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
    }
    args_task_a.log_dir = log_dir
    args_task_b = args_task_a.copy()
    args_task_c = args_task_a.copy()
    args_task_b["task"] = "b"
    args_task_c["task"] = "c"
    args_task_a = Namespace(**args_task_a)
    args_task_b = Namespace(**args_task_b)
    args_task_c = Namespace(**args_task_c)
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
    }
    args_wrapper = Namespace(**args_wrapper)

    # Setup WandB logging
    wandb_logger = helpers.setup_wandb(args_wrapper)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Run Arguments are (Task A):\n {pprint.pformat(vars(args_task_a), sort_dicts=False)}"
    )
    logger.info(
        f"Run Arguments are (Task B):\n {pprint.pformat(vars(args_task_b), sort_dicts=False)}"
    )
    logger.info(
        f"Run Arguments are (Task C):\n {pprint.pformat(vars(args_task_c), sort_dicts=False)}"
    )
    seed_everything(args_task_a.random_seed)

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

    download_links = helpers.get_model_download_links(model_name)
    model_paths = list()
    api = wandb.Api()
    for link in download_links:
        artifact = api.artifact(link)
        model_paths.append(artifact.download())
    args_task_a.ckpt_path = model_paths[0]
    args_task_b.ckpt_path = model_paths[1]
    args_task_c.ckpt_path = model_paths[2]

    lightning_callbacks = helpers.get_lightning_callbacks(args_task_a)
    wrapper_model = WrapperTransformerModule(args_task_a, args_task_b, args_task_c)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args_wrapper.num_epoch,
    )

    # Train
    if args_wrapper.train:
        trainer.fit(model=wrapper_model, datamodule=data_module_task_a)

    # Eval TODO: handle checkpoint loading
    if args_wrapper.eval:
        trainer.test(model=wrapper_model, datamodule=data_module_task_a, ckpt_path="best")


if __name__ == "__main__":
    main("roberta-base")
