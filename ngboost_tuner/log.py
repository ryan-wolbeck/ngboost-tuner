import time
import logging
from typing import Any


# creates a new logger as well as assigns this logger to the root logger in the logging package
# this means that you can import logging, use logging.info("test"), and you will use the logger
# that is defined below
def new_logger(args: Any) -> logging.Logger:
    logger = logging.getLogger("root")

    logger.setLevel(logging.INFO)

    log_format = "%(asctime)s.%(msecs)06f"

    if hasattr(args, "debug") and args.debug:
        logger.setLevel(logging.DEBUG)
        log_format += "\t%(filename)s\t%(lineno)d\t%(funcName)s"

    log_format += f"\t%(message)s"
    date_format = "%Y-%m-%dT%H:%M:%S"

    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.root = logger
    return logger
