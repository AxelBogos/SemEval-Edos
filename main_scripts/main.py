import logging
import pprint

from dotenv import load_dotenv

from src.utils import defines, helpers
from src.utils.args import parse_args


def main():
    load_dotenv(defines.DOTENV_FILE)
    log_dir = helpers.make_log_dir()
    helpers.setup_python_logging(log_dir)
    args = parse_args()
    args.log_dir = log_dir

    logger = logging.getLogger(__name__)
    logger.info(f"Run Arguments are:\n {pprint.pformat(vars(args),sort_dicts=False)}")


if __name__ == "__main__":
    main()
