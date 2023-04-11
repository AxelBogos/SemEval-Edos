import logging
import pprint
from argparse import Namespace

from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything

from src.utils import defines, helpers
from src.utils.args import parse_args


def main():
    """The main function is the entry point of the program.

    It initializes all necessary variables and objects, then starts training.
    """

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
        "model": "distilroberta-base",
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

    optimizer_task_a = helpers.get_optimizer(args_task_a)
    optimizer_task_b = helpers.get_optimizer(args_task_b)
    optimizer_task_c = helpers.get_optimizer(args_task_c)

    model_task_a = helpers.get_model(args_task_a, optimizer_task_a)
    model_task_b = helpers.get_model(args_task_b, optimizer_task_b)
    model_task_c = helpers.get_model(args_task_c, optimizer_task_c)

    lightning_callbacks_task_a = helpers.get_lightning_callbacks(args_task_a)
    lightning_callbacks_task_b = helpers.get_lightning_callbacks(args_task_b)
    lightning_callbacks_task_c = helpers.get_lightning_callbacks(args_task_c)

    trainer_task_a = Trainer(
        logger=logger,
        callbacks=lightning_callbacks_task_a,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_a.num_epoch,
    )
    trainer_task_b = Trainer(
        logger=logger,
        callbacks=lightning_callbacks_task_b,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_b.num_epoch,
    )
    trainer_task_c = Trainer(
        logger=logger,
        callbacks=lightning_callbacks_task_c,
        accelerator="auto",
        devices="auto",
        max_epochs=args_task_c.num_epoch,
    )
    trainer_task_a.load

    # Setup WandB logging
    # wandb_logger = helpers.setup_wandb(args)
    # Build model

    # Train
    # if args.train:
    #     trainer.fit(model=model, datamodule=data_module)
    #
    # # Eval TODO: handle checkpoint loading
    # if args.eval:
    #     trainer.test(model=model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
