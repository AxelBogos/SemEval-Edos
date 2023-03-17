import logging
import pprint

from dotenv import load_dotenv
from pytorch_lightning import Trainer, seed_everything

from src.utils import defines, helpers
from src.utils.args import parse_args


def main():
    load_dotenv(defines.DOTENV_FILE)
    log_dir = helpers.make_log_dir()
    helpers.setup_python_logging(log_dir)
    args = parse_args()
    args.log_dir = log_dir

    logger = logging.getLogger(__name__)
    logger.info(f"Run Arguments are:\n {pprint.pformat(vars(args), sort_dicts=False)}")
    seed_everything(args.random_seed)
    data_module = helpers.get_data_module(args)
    data_module.setup()
    args.len_vocab = len(data_module.vocab)
    args.num_target_class = data_module._num_classes
    args.pad_idx = data_module.pad_idx
    optimizer = helpers.get_optimizer(args)
    scheduler = helpers.get_scheduler(args)
    model = helpers.get_model(args, optimizer, scheduler)
    wandb_logger = helpers.setup_wandb(args)
    lightning_callbacks = helpers.get_lightning_callbacks(args)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=lightning_callbacks,
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epoch,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
