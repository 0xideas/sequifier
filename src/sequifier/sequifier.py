from argparse import ArgumentParser
from typing import Any

import numpy as np
from beartype import beartype

from sequifier.hyperparameter_search import hyperparameter_search
from sequifier.infer import infer
from sequifier.make import make
from sequifier.preprocess import preprocess
from sequifier.train import train


@beartype
def build_args_config(args: Any) -> dict[str, Any]:
    """
    Build configuration dictionary from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dictionary containing configuration options.
    """
    args_config = {
        k: v for k, v in vars(args).items() if v is not None and k != "randomize"
    }
    if args.command != "make":
        if args.randomize:
            seed = np.random.choice(np.arange(int(1e9)))
            args_config["seed"] = seed
        elif args.command in ["train", "infer"] and args.seed:
            args_config["seed"] = args.seed
        else:
            args_config["seed"] = 1010

    if "selected_columns" in args_config:
        if args_config["selected_columns"] == "None":
            args_config["selected_columns"] = None
        else:
            args_config["selected_columns"] = (
                args_config["selected_columns"].replace(" ", "").split(",")
            )

    return args_config


@beartype
def setup_parser() -> ArgumentParser:
    """
    Set up the argument parser for the command-line interface.

    Returns:
        Configured ArgumentParser object.
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_make = subparsers.add_parser("make", help="Set up sequifier project")
    parser_make.add_argument("project_name", type=str)
    parser_preprocess = subparsers.add_parser(
        "preprocess", help="Run the preprocessing step"
    )
    parser_train = subparsers.add_parser("train", help="Run the training step")
    parser_infer = subparsers.add_parser("infer", help="Run the inference step")

    parser_hyperparameter_search = subparsers.add_parser(
        "hyperparametersearch", help="Run hyperparamter search"
    )

    for subparser in [
        parser_preprocess,
        parser_train,
        parser_infer,
        parser_hyperparameter_search,
    ]:
        subparser.add_argument(
            "--config-path",
            type=str,
            help="File path to config for current processing step",
        )

        if subparser != parser_hyperparameter_search:
            subparser.add_argument("-r", "--randomize", action="store_true")
            subparser.add_argument("-sc", "--selected-columns", type=str)
            subparser.add_argument("-dp", "--data-path", type=str)

    for subparser in [parser_train, parser_infer, parser_hyperparameter_search]:
        subparser.add_argument("-ddcp", "--ddconfig-path", type=str)
        subparser.add_argument("-op", "--on-unprocessed", action="store_true")

    parser_train.add_argument("-mn", "--model-name", type=str)
    parser_train.add_argument("-s", "--seed", type=int)
    parser_infer.add_argument("-imp", "--model-path", type=str)
    parser_infer.add_argument("-s", "--seed", type=int)

    return parser


def main() -> None:
    """
    Main function to run the Sequifier CLI.
    """
    parser = setup_parser()
    args = parser.parse_args()

    if args.command != "hyperparametersearch":
        args_config = build_args_config(args)

        if args.command == "make":
            make(args)
        elif args.command == "preprocess":
            preprocess(args, args_config)
        elif args.command == "train":
            train(args, args_config)
        elif args.command == "infer":
            infer(args, args_config)
    elif args.command == "hyperparametersearch":
        hyperparameter_search(args.config_path, args.on_unprocessed)


if __name__ == "__main__":
    main()
