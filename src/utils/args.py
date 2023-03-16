import argparse
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
        default=None,
        type=str,
        help="Name of the run to load weights from",
    )

    parser.add_argument("--batch_size", default=64, type=int, help="Mini batch size")

    parser.add_argument("--num_epoch", default=80, type=int, help="Number of epoch to train")

    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.",
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
    return args
