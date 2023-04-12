import logging
import pprint

from dotenv import load_dotenv
from lightning import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner

# from pytorch_lightning import Trainer, seed_everything
from sklearn.utils.class_weight import compute_class_weight

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
    args = parse_args()
    args.log_dir = log_dir

    # Setup WandB logging
    wandb_logger = helpers.setup_wandb(args)

    logger = logging.getLogger(__name__)
    logger.info(f"Run Arguments are:\n {pprint.pformat(vars(args), sort_dicts=False)}")
    seed_everything(args.random_seed)

    # Build data module
    data_module = helpers.get_data_module(args)
    data_module.setup()
    args.num_target_class = data_module._num_classes
    args.len_train_loader = len(data_module.train_dataloader())
    if args.architecture == "lstm":
        args.len_vocab = len(data_module.vocab)
        args.pad_idx = data_module.pad_idx

    # Build model
    optimizer = helpers.get_optimizer(args)
    model = helpers.get_model(args, optimizer)
    lightning_callbacks = helpers.get_lightning_callbacks(args)
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args.num_epoch,
    )
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=data_module)
    new_lr = lr_finder.suggestion()
    model.learning_rate = new_lr
    model.hparams.learning_rate = new_lr
    # Train
    if args.train:
        trainer.fit(model=model, datamodule=data_module)

    # Eval TODO: handle checkpoint loading
    if args.eval:
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
