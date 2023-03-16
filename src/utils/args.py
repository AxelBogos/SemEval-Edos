# Imports
import argparse
import logging
import os.path

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
        help="Model name; available models: {lstm, TODO}",
    )

    parser.add_argument(
        "--task",
        default="a",
        type=str,
        help="Task name; available tasks: {a, b, c}",
    )

    parser.add_argument(
        "--load_weights_from",
        type=str,
        help="Name of the run to load weights from",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Name of the run",
    )

    parser.add_argument("--batch_size", default=64, type=int, help="Mini batch size")

    parser.add_argument(
        "--num_epoch", default=80, type=int, help="Number of epoch to train. Geirhos uses 80."
    )

    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.",
    )

    # I/O dirs
    parser.add_argument(
        "--logs_dir", default=defines.LOG_DIR, help="Directory to write trained models to"
    )

    parser.add_argument(
        "--data_dir", default=defines.DATA_DIR, help="Root directory with all datasets"
    )

    parser.add_argument(
        "--random_seed", default=3454572, type=int, help="Random seed to use throughout run."
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments are: {args}")
    return args
