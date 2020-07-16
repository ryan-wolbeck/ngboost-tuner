import os
import sys
import argparse
from ngboost_tuner.tune import run as run_tune


def input_file(parser, path):
    if path is None:
        return sys.stdin
    if not os.path.exists(path):
        parser.error(f"{path} path does not exist")
        sys.exit(1)
    return open(path, "r")


def build_cli():
    root = argparse.ArgumentParser(prog="ngboost_tuner")
    root.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode; defaults to false",
    )

    subparsers = root.add_subparsers(title="commands", dest="cmd")

    tune = subparsers.add_parser("tune")
    tune.set_defaults(func=run_tune)
    tune.add_argument(
        "-i",
        "--input",
        "--input-file",
        type=lambda path: input_file(tune, path),
        default=os.getenv("INPUT_FILE"),
        help="Input file data; defaults to $INPUT_FILE if not set",
    )
    tune.add_argument(
        "-t",
        "--target",
        type=str,
        default=os.getenv("TARGET"),
        help=f"Target variable (predicted variable). Default value: {os.getenv('TARGET')}",
    )
    tune.add_argument(
        "-id",
        "--id-key",
        type=str,
        default=os.getenv("ID"),
        help=f"ID to consider for splits to prevent leakage. Default {os.getenv('ID')}",
    )
    tune.add_argument(
        "-c",
        "--column",
        action="append",
        type=list,
        default=list(os.getenv("TRAIN_COLUMNS").split(",")),
        help=f"The full list of columns: Defaults to {os.getenv('TRAIN_COLUMNS')}",
    )
    tune.add_argument(
        "-ef",
        "--evaluation-fraction",
        type=float,
        default=0.2,
        help="Proportion of loadnums used for evaluation .2 is 20 percent of training leaving 80 percent train, 10 percent test, 10 percent validation. Default = .2",
    )

    return root
