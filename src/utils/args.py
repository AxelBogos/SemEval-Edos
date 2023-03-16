import argparse
import logging
from pathlib import Path

from src.utils import defines


def parse_args():
    """The parse_args function is used to parse the command line arguments. It takes no arguments
    and returns an object containing all of the parsed arguments. The returned object has
    attributes corresponding to each of the command line flags, e.g., args.train will be True if.

    --train was passed as a flag.

    :return: An object with the following attributes:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",  # 'store_false' if want default to false
        help="Do Train",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",  # 'store_false' if want default to false
        help="Do Evaluation",
    )

    parser.add_argument(
        "--model",
        default="bilstm",
        type=str,
        help="Model name; available models: {'gnb','bilstm','distillbert'}",
    )

    parser.add_argument(
        "--task",
        default="a",
        type=str,
        help="Task name; available tasks: {a, b, c}",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="Optimizer name; available tasks: {Adam, AdamW, SGD}",
    )

    parser.add_argument(
        "--load_weights_from",
        default=None,
        type=str,
        help="Name of the run to load weights from",
    )

    parser.add_argument(
        "--preprocessing_mode",
        default="standard",
        type=str,
        help="Type of preprocessing to apply. Choices as {'standard', 'none'}. Standard preprocessing is not meant for transformers models.",
    )

    parser.add_argument("--batch_size", default=64, type=int, help="Mini batch size")

    parser.add_argument("--max_length", default=128, type=int, help="Max Sequence Length")

    parser.add_argument("--num_epoch", default=80, type=int, help="Number of epoch to train")

    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.",
    )

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number workers",
    )

    # I/O dirs
    parser.add_argument(
        "--logs_dir",
        default=defines.LOG_DIR,
        type=Path,
        help="Directory to write trained models to",
    )

    parser.add_argument(
        "--raw_data_dir",
        default=defines.RAW_DATA_DIR,
        type=Path,
        help="Raw data directory with all datasets",
    )
    parser.add_argument(
        "--interim_data_dir",
        default=defines.INTERIM_DATA_DIR,
        type=Path,
        help="Interim data directory with all datasets",
    )
    parser.add_argument(
        "--processed_data_dir",
        default=defines.PROCESSED_DATA_DIR,
        type=Path,
        help="Processed data directory with all datasets",
    )

    parser.add_argument(
        "--random_seed", default=3454572, type=int, help="Random seed to use throughout run."
    )

    args = parser.parse_args()
    if not _args_sanity_check(args):
        exit()
    return args


def _args_sanity_check(args) -> bool:

    """The _args_sanity_check function performs a series of checks on the command line arguments
    passed to the main function. It returns True if all checks pass, and False otherwise.

    :param args: Pass in the command line arguments
    :return: True if all the arguments are valid
    """
    logger = logging.getLogger(__name__)

    if not args.train and not args.eval:
        logger.error("Neither Training or Eval mode selected.")
        return False
    if args.task not in defines.VALID_TASK_CHOICES:
        logger.error(
            f"args.task = {args.task} is not a valid choice. Valid Choices: {defines.VALID_TASK_CHOICES}"
        )
        return False
    if args.model not in defines.VALID_MODEL_CHOICES:
        logger.error(
            f"args.model = {args.model} is not a valid choice. Valid Choices: {defines.VALID_MODEL_CHOICES}"
        )
        return False
    if args.preprocessing_mode not in defines.VALID_PREPROCESSING_MODE:
        logger.error(
            f"args.preprocessing_mode = {args.preprocessing_mode} is not a valid choice. Valid Choices: {defines.VALID_PREPROCESSING_MODE}"
        )
        return False
    if args.optimizer not in defines.VALID_OPTIMIZERS:
        logger.error(
            f"args.preprocessing_mode = {args.optimizer} is not a valid choice. Valid Choices: {defines.VALID_OPTIMIZERS}"
        )
        return False
    if not Path(args.raw_data_dir).is_dir():
        logger.error(f"args.raw_data_dir = {args.raw_data_dir} is not a dir.")
        return False
    if not Path(args.interim_data_dir).is_dir():
        logger.error(f"args.interim_data_dir = {args.interim_data_dir} is not a dir.")
        return False
    if not Path(args.processed_data_dir).is_dir():
        logger.error(f"args.processed_data_dir = {args.processed_data_dir} is not a dir.")
        return False
    return True
