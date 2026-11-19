from argparse import ArgumentParser, ArgumentTypeError
from typing import Any

import yaml

from sequifier.docs import docs
from sequifier.hyperparameter_search import hyperparameter_search
from sequifier.infer import infer
from sequifier.make import make
from sequifier.preprocess import preprocess
from sequifier.train import train
from sequifier.typechecking import beartype
from sequifier.visualize_training import visualize_training


class ConfigArgumentParser(ArgumentParser):
    """Parse only explicitly declared command options."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def parse_seed(value: str) -> int | str:
    if value == "random":
        return value
    try:
        return int(value)
    except ValueError as error:
        raise ArgumentTypeError("seed must be an integer or random") from error


def parse_boolean(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ArgumentTypeError("boolean overrides must be true or false")
    return value == "true"


@beartype
def build_args_config(args: Any) -> dict[str, Any]:
    """Build config overrides from parsed CLI args."""

    args_config = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["command", "config_path"]
    }
    if "selected_columns" in args_config:
        if (
            len(args_config["selected_columns"]) == 1
            and args_config["selected_columns"][0] == "None"
        ):
            args_config["selected_columns"] = None

    if "input_columns" in args_config:
        if (
            len(args_config["input_columns"]) == 1
            and args_config["input_columns"][0] == "None"
        ):
            args_config["input_columns"] = None

    return args_config


@beartype
def setup_parser() -> ArgumentParser:
    """Build the sequifier CLI parser."""
    parser = ConfigArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_make = subparsers.add_parser("make", help="Set up sequifier project")
    parser_make.add_argument("project_name", type=str)
    parser_docs = subparsers.add_parser("docs", help="Print sequifier documentation")
    parser_docs.add_argument(
        "topic",
        nargs="?",
        choices=("train", "preprocess", "infer", "policy"),
        help="Show documentation for one command (default: all documentation)",
    )
    parser_preprocess = subparsers.add_parser(
        "preprocess", help="Run the preprocessing step"
    )
    parser_train = subparsers.add_parser(
        "train",
        help="Train supervised models or ordered imitation/IQL/PPO/DPO/GRPO policy phases",
    )
    parser_infer = subparsers.add_parser("infer", help="Run the inference step")
    parser_hyperparameter_search = subparsers.add_parser(
        "hyperparameter-search", help="Run hyperparamter search"
    )
    parser_visualize_training = subparsers.add_parser(
        "visualize-training", help="Visualize training losses"
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

    for subparser in [parser_preprocess, parser_train, parser_infer]:
        subparser.add_argument(
            "--seed", type=parse_seed, help="Integer run seed, or random"
        )

    parser_preprocess.add_argument(
        "--preprocessing-data-path",
        type=str,
    )
    parser_infer.add_argument("--data-path", type=str)
    parser_infer.add_argument(
        "--input-columns",
        type=str,
        nargs="+",
        help="Space-separated list of input columns (e.g. --input-columns col1 col2)",
    )
    parser_infer.add_argument("--metadata-config-path", type=str)

    for subparser in [parser_train, parser_infer, parser_hyperparameter_search]:
        subparser.add_argument("--skip-metadata", action="store_true")

    parser_preprocess.add_argument(
        "--selected-columns",
        type=str,
        nargs="+",
        help="Space-separated list of selected columns",
    )
    parser_train.add_argument("--model-name", type=str)
    for subparser in (parser_train, parser_infer, parser_preprocess):
        subparser.add_argument("--run", type=yaml.safe_load)
    parser_train.add_argument("--training-plan", type=yaml.safe_load)

    parser_infer.add_argument("--model-paths", type=str, nargs="+")
    parser_infer.add_argument("--enforce-determinism", type=parse_boolean, default=None)
    parser_infer.add_argument(
        "--action-selection", choices=("deterministic", "stochastic")
    )
    parser_infer.add_argument("--task", choices=("forecast", "policy", "embedding"))
    parser_infer.add_argument("--dataset", type=str)
    parser_infer.add_argument("--part", type=str)
    parser_infer.add_argument("--model-interface", type=str)

    parser_visualize_training.add_argument(
        "models",
        type=str,
        help=(
            "Model name, hyperparameter search name, comma-separated names, "
            "or path to txt file containing model names"
        ),
    )
    parser_visualize_training.add_argument(
        "--log-scale",
        action="store_true",
        help="Use a logarithmic scale on the y-axis",
    )
    parser_visualize_training.add_argument(
        "--bucket-training-batches",
        type=int,
        help="Bucket the training batches by taking the average loss over a specified number of batches. Must be a multiple of the logged batch interval.",
    )

    parser_visualize_training.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Path to the project root directory.",
    )

    return parser


@beartype
def main() -> None:
    """
    Main function to run the Sequifier CLI.
    """
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "docs":
        docs(args.topic)
        return

    if args.command != "hyperparameter-search":
        args_config = build_args_config(args)

        if args.command == "make":
            make(args)
        elif args.command == "preprocess":
            preprocess(args, args_config)
        elif args.command == "train":
            train(args, args_config)
        elif args.command == "infer":
            infer(args, args_config)
        elif args.command == "visualize-training":
            visualize_training(args)
    elif args.command == "hyperparameter-search":
        hyperparameter_search(args.config_path, args.skip_metadata)


if __name__ == "__main__":
    main()
